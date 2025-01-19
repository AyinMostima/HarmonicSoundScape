import librosa
import librosa.feature
from pandas import read_pickle
from pyAudioAnalysis import audioTrainTest as aT
import os
from maad import sound, features
from maad.util import date_parser
import pandas as pd
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import params
import csv
import random
from madmom.features.key import CNNKeyRecognitionProcessor
from madmom.features.tempo import TempoEstimationProcessor
from madmom.features.beats import RNNBeatProcessor
from madmom.features.onsets import RNNOnsetProcessor, OnsetPeakPickingProcessor
import time
import datetime
import json
from maad import sound, util, rois
from maad.util import power2dB
from scipy.signal import butter, filtfilt
from scipy.linalg import circulant
import numba
import time
import contextlib

# 禁止提醒
import warnings
from scipy.io.wavfile import WavFileWarning
from sklearn.exceptions import InconsistentVersionWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=WavFileWarning)

SPECTRAL_FEATURES = ['MEANf', 'VARf', 'SKEWf', 'KURTf', 'NBPEAKS', 'LEQf',
                     'ENRf', 'BGNf', 'SNRf', 'Hf', 'EAS', 'ECU', 'ECV', 'EPS', 'EPS_KURT', 'EPS_SKEW', 'ACI',
                     'NDSI', 'rBA', 'AnthroEnergy', 'BioEnergy', 'BI', 'ROU', 'ADI', 'AEI', 'LFC', 'MFC', 'HFC',
                     'ACTspFract', 'ACTspCount', 'ACTspMean', 'EVNspFract', 'EVNspMean', 'EVNspCount',
                     'TFSD', 'H_Havrda', 'H_Renyi', 'H_pairedShannon', 'H_gamma', 'H_GiniSimpson', 'RAOQ',
                     'AGI', 'ROItotal', 'ROIcover'
                     ]

TEMPORAL_FEATURES = ['ZCR', 'MEANt', 'VARt', 'SKEWt', 'KURTt',
                     'LEQt', 'BGNt', 'SNRt', 'MED', 'Ht', 'ACTtFraction', 'ACTtCount',
                     'ACTtMean', 'EVNtFraction', 'EVNtMean', 'EVNtCount'
                     ]

# 和弦监测代码
# 和现识别函数准备
# 和弦标签定义
chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
minor_chords = [s + 'm' for s in chroma_labels]  # 小三和弦
major_chords = chroma_labels  # 大三和弦
chord_labels = major_chords + minor_chords  # 共24个和弦


# 色度特征提取
def highpass_filter(x, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency

    # Design the highpass filter
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    # Apply the filter to the signal
    y = filtfilt(b, a, x)
    return y


def lowpass_filter(x, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency

    # Design the lowpass filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter to the signal
    y = filtfilt(b, a, x)
    return y


def compute_chromagram_from_filename(fn_wav, Fs=None, N=4096, H=2048, gamma=None, version='STFT', ifroi=False,
                                     frepass=False):
    # 加载音频文件
    x, Fs = librosa.load(fn_wav, sr=Fs)
    x_dur = x.shape[0] / Fs

    # 如果需要低通滤波
    if frepass:
        x = highpass_filter(x, cutoff=2000, fs=Fs)
        # x = lowpass_filter(x, cutoff=10000, fs=Fs)

    # 如果需要ROI，则提取ROI部分
    if ifroi:

        # 计算功率谱
        Sxx_power, tn, fn, ext = sound.spectrogram(x=x, fs=Fs, nperseg=1024, noverlap=1024 // 2, display=False)
        dB_max = 120
        Sxx_power_noNoise = sound.median_equalizer(Sxx=Sxx_power, **{'extent': ext})
        Sxx_db_noNoise = power2dB(Sxx_power_noNoise)
        # # 生成掩码并选择ROI
        Sxx_db_noNoise_smooth = sound.smooth(Sxx=Sxx_db_noNoise, std=0.3, **{'vmin': 0, 'vmax': dB_max, 'extent': ext},
                                             display=False)
        im_mask = rois.create_mask(im=Sxx_db_noNoise_smooth, mode_bin='relative', bin_std=8, bin_per=0.5, verbose=False,
                                   display=False)
        im_rois, df_rois = rois.select_rois(im_bin=im_mask, min_roi=25)

        # 获取ROI时间范围，假设df_rois中的'begin'和'end'是ROI的开始和结束时间
        # 使用 min_x 和 max_x 来获取时间范围
        roi_time_start = df_rois['min_x'].values
        roi_time_end = df_rois['max_x'].values
        roi_audio = []

        for start, end in zip(roi_time_start, roi_time_end):
            # 转换为样本索引
            start_idx = librosa.time_to_samples(start, sr=Fs)
            end_idx = librosa.time_to_samples(end, sr=Fs)
            roi_audio.append(x[start_idx:end_idx])  # 提取对应时间段的音频

        # 检查 ROI 音频长度，若小于 5，则跳出并返回空
        if len(roi_audio) < 3:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])  # 返回空的色度图和其他信息

        # 将所有ROI部分合并为一个信号
        x_roi = np.concatenate(roi_audio)
    else:
        # 如果不使用ROI，直接使用整个音频信号
        x_roi = x

    # 根据选择的版本计算色度图
    if version == 'STFT':
        # 使用STFT计算色度图
        X = librosa.stft(x_roi, n_fft=N, hop_length=H, pad_mode='constant', center=True)
        if gamma is not None:
            X = np.log(1 + gamma * np.abs(X) ** 2)
        else:
            X = np.abs(X) ** 2
        X_chroma = librosa.feature.chroma_stft(S=X, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)

    elif version == 'CQT':
        # 使用CQT计算色度图
        X_chroma = librosa.feature.chroma_cqt(y=x_roi, sr=Fs, hop_length=H, norm=None)

    elif version == 'IIR':
        # 使用滤波器组计算色度图
        X = librosa.iirt(y=x_roi, sr=Fs, win_length=N, hop_length=H, center=True, tuning=0.0)
        if gamma is not None:
            X = np.log(1.0 + gamma * X)
        X_chroma = librosa.feature.chroma_cqt(C=X, bins_per_octave=12, n_octaves=7, fmin=librosa.midi_to_hz(24),
                                              norm=None)

    # 计算色度图的采样率
    Fs_X = Fs / H
    return X_chroma, Fs_X, x_roi, Fs, x_dur, X


# 第一次识别，和弦模板(前处理)

def get_chord_labels(ext_minor='m', nonchord=False):
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_labels_maj = chroma_labels
    chord_labels_min = [s + ext_minor for s in chroma_labels]
    chord_labels = chord_labels_maj + chord_labels_min
    if nonchord is True:
        chord_labels = chord_labels + ['N']
    return chord_labels


# def generate_chord_templates(nonchord=False):
#     template_cmaj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).T
#     template_cmin = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]).T
#     num_chord = 24
#     if nonchord:
#         num_chord = 25
#     chord_templates = np.ones((12, num_chord))
#     for shift in range(12):
#         chord_templates[:, shift] = np.roll(template_cmaj, shift)
#         chord_templates[:, shift + 12] = np.roll(template_cmin, shift)
#     return chord_templates

def generate_chord_templates(a=0.9, nonchord=False):
    # 定义每个和弦的部分音及其对应的和声衰减
    # 和弦C的大调和声模板示例
    # thC = (1 + a + a^3 + a^7, 0, 0, 0, a^4, 0, 0, a^2 + a^5, 0, 0, a^6, 0).T
    def create_harmonic_template(a):

        th = np.zeros(12)
        # 部分音的位置相对于基音C
        # 以C为例，前八个部分音对应的音高位置
        harmonics = [0, 0, 7, 0, 4, 7, 11, 0]  # 对应C, C, G, C, E, G, B, C
        # 能量衰减指数对应部分音的顺序
        decay_exponents = [0, 1, 3, 7, 4, 5, 6]
        # 对应每个部分音的位置和能量
        for i, (pitch, exp) in enumerate(zip(harmonics, decay_exponents)):
            if pitch != 0:  # 忽略0位置（C）
                th[pitch % 12] += a ** exp
            else:
                th[0] += a ** 0  # 基音C
        return th

    # 创建大调和小调的基本模板
    # 大调和弦包含根音、第三度和第五度
    # 小调和弦包含根音、降第三度和第五度
    def create_chord_template(base_chroma, mode='major'):
        chord_th = np.zeros(12)
        # 定义大调和小调的第三度和第五度
        if mode == 'major':
            third = (base_chroma + 4) % 12
        elif mode == 'minor':
            third = (base_chroma + 3) % 12
        else:
            raise ValueError("mode must be 'major' or 'minor'")
        fifth = (base_chroma + 7) % 12
        # 叠加根音、第三度和第五度的和声模板
        chord_th += np.roll(create_harmonic_template(a), base_chroma)
        chord_th += np.roll(create_harmonic_template(a), third)
        chord_th += np.roll(create_harmonic_template(a), fifth)
        return chord_th

    num_chord = 24  # 12大调 + 12小调
    if nonchord:
        num_chord += 1  # 增加一个非和弦类别
    chord_templates = np.zeros((12, num_chord))

    # 生成12个大调和12个小调的和弦模板
    for root in range(12):
        # 大调和弦
        major_template = create_chord_template(root, mode='major')
        chord_templates[:, root] = major_template
        # 小调和弦
        minor_template = create_chord_template(root, mode='minor')
        chord_templates[:, root + 12] = minor_template

    # 如果需要，添加非和弦类别（全1或其他定义）
    if nonchord:
        chord_templates[:, -1] = 1  # 这里设为全1向量，可根据需要调整

    return chord_templates


def chord_recognition_template(X_chroma, norm_sim='1', nonchord=False):
    chord_templates = generate_chord_templates(nonchord=nonchord)
    chord_sim = np.dot(chord_templates.T, X_chroma)  # Compute similarity
    if norm_sim == '1':
        chord_sim = chord_sim / np.linalg.norm(chord_sim, axis=0)  # Normalize by L2 norm
    elif norm_sim == 'max':
        chord_sim = chord_sim / np.max(np.abs(chord_sim), axis=0)  # Normalize by max value
    # 计算每个时间帧对应的最匹配和弦
    chord_max = np.argmax(chord_sim, axis=0)

    return chord_sim, chord_max


# 第二次识别，马尔可夫(后处理)
def matrix_circular_mean(A):
    N = A.shape[0]
    A_shear = np.zeros((N, N))
    for n in range(N):
        A_shear[:, n] = np.roll(A[:, n], -n)
    circ_sum = np.sum(A_shear, axis=1)
    A_mean = circulant(circ_sum) / N
    return A_mean


def matrix_chord24_trans_inv(A):
    A_ti = np.zeros(A.shape)
    A_ti[0:12, 0:12] = matrix_circular_mean(A[0:12, 0:12])
    A_ti[0:12, 12:24] = matrix_circular_mean(A[0:12, 12:24])
    A_ti[12:24, 0:12] = matrix_circular_mean(A[12:24, 0:12])
    A_ti[12:24, 12:24] = matrix_circular_mean(A[12:24, 12:24])
    return A_ti


def uniform_transition_matrix(p=0.01, N=24):
    off_diag_entries = (1 - p) / (N - 1)  # rows should sum up to 1
    A = off_diag_entries * np.ones([N, N])
    np.fill_diagonal(A, p)
    return A


def viterbi_log_likelihood(A, C, B_O):
    I = A.shape[0]  # Number of states
    N = B_O.shape[1]  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_O_log = np.log(B_O + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N - 1)).astype(np.int32)
    D_log[:, 0] = C_log + B_O_log[:, 0]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n - 1]
            D_log[i, n] = np.max(temp_sum) + B_O_log[i, n]
            E[i, n - 1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N - 2, -1, -1):
        S_opt[n] = E[int(S_opt[n + 1]), n]

    # Matrix representation of result
    S_mat = np.zeros((I, N)).astype(np.int32)
    for n in range(N):
        S_mat[S_opt[n], n] = 1

    return S_mat, S_opt, D_log, E


# Load transition matrix estimated on the basis of the Beatles collection
fn_csv = os.path.join('transitionMatrix_Beatles.csv')
A_est_df = pd.read_csv(fn_csv, delimiter=';')
A_est = A_est_df.to_numpy('float64')
A_ti = matrix_chord24_trans_inv(A_est)
A_un = uniform_transition_matrix(p=0.5)


def chordrecog(file_name, N=4096, H=2048, ifroi=False, transma='un'):
    X_chroma, Fs_X, x, Fs, x_dur, X_stft = compute_chromagram_from_filename(file_name, N=N, H=H, gamma=0.5,
                                                                            version='STFT', frepass=True, ifroi=ifroi)
    if X_chroma.all() == None:
        return np.array([]), np.array([])  # 返回空的色度图和其他信息
    else:

        chord_sim, _ = chord_recognition_template(X_chroma, norm_sim='1', nonchord=False)
        if transma == 'un':
            A = A_un
        if transma == 'beatles':
            A = A_ti
        C = 1 / 24 * np.ones((1, 24))
        B_O = chord_sim
        chord_HMM, _, _, _ = viterbi_log_likelihood(A, C, B_O)
        chord_max = np.argmax(chord_HMM, axis=0)

        return chord_HMM, chord_max, X_chroma, Fs_X


# 去重的和弦列表：前后相同的和弦只保留一次
def reduce_chord_sequence(chord_list):
    reduced_list = [chord_list[0]]  # 初始化第一个和弦
    for chord in chord_list[1:]:
        if chord != reduced_list[-1]:  # 如果当前和弦与上一个不同，则添加到列表
            reduced_list.append(chord)
    return reduced_list


# 创建单样本的 DataFrame
def process_single_sample(chord_HMM, X_chroma, chord_max, chord_labels, Fs_X):
    if X_chroma.all() == None:
        return pd.DataFrame({
            'chordhmm': np.array([]),  # 原始和弦索引
            'chroma': np.array([]),  # 色度图矩阵
            'chord': np.array([]),  # 去重后的和弦序列
            'chordtime': np.array([])  # 时间范围和弦标签
        })

    else:

        # 确保 chord_HMM 是整数
        chord_HMM = np.array(chord_HMM, dtype=int)

        # 转换为和弦标签
        chord_labels_HMM = [chord_labels[i] for i in chord_max]

        # 计算时间范围并生成 chordtime
        time_frames = np.arange(X_chroma.shape[1]) / Fs_X

        chordtime = []
        start_time = time_frames[0]
        current_chord = chord_labels_HMM[0]
        for i in range(1, len(chord_labels_HMM)):
            if chord_labels_HMM[i] != current_chord:
                end_time = time_frames[i]
                chordtime.append(f"{{{round(start_time, 2)}-{round(end_time, 2)}: {current_chord}}}")
                start_time = time_frames[i]
                current_chord = chord_labels_HMM[i]
        # 结束时，补上最后一段
        if len(chord_labels_HMM) > 0:
            chordtime.append(f"{{{round(start_time, 2)}-{round(time_frames[-1], 2)}: {current_chord}}}")

        # 生成去重的和弦序列
        chordsequence = reduce_chord_sequence(chord_labels_HMM)

        # 构造单行 DataFrame
        df_sample = pd.DataFrame({
            'chordhmm': [chord_HMM.tolist()],  # 原始和弦索引
            'chroma': [X_chroma.tolist()],  # 色度图矩阵
            'chord': [chordsequence],  # 去重后的和弦序列
            'chordtime': [chordtime]
        })

        return df_sample


# 失真检验
def detect_clipping(file_wav, threshold=0.99, clip_percentage=0.2):
    """
    检测音频文件中的失真（削波）情况。

    参数:
    file_wav (str): 音频文件路径
    threshold (float): 削波检测阈值，默认值为0.99

    返回:
    bool: 如果检测到失真返回True，否则返回False
    """
    # 加载音频文件
    y, sr = librosa.load(file_wav, sr=None)

    # 检测信号饱和或削波失真
    clipping_indices = np.where(np.abs(y) > threshold)[0]

    # 检查失真占比是否超过指定比例
    if len(clipping_indices) / len(y) > clip_percentage:
        return True
    else:
        return False


def detect_music_and_speech(audio_file):
    # 加载音频文件
    signal, sr = librosa.load(audio_file, sr=None)  # 使用音频本身的采样率
    signal = signal - np.mean(signal)

    # 使用pyAudioAnalysis进行音乐和语音检测
    result_index, probabilities, classes = aT.file_classification(audio_file,
                                                                  "pretrained_models/data/models/svm_rbf_sm", "svm")

    # 获取最可能的类别索引
    result_index = int(result_index)  # 确保索引是整数

    # 解析返回结果
    is_music = classes[result_index] == 'music'
    is_speech = classes[result_index] == 'speech'
    music_probability = probabilities[classes.index('music')]
    speech_probability = probabilities[classes.index('speech')]

    return is_music, is_speech


# 读取类名
def class_names(class_map_csv):
    """Read the class name definition file and return a list of strings."""
    with open(class_map_csv) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header
        return np.array([display_name for (_, _, display_name) in reader])


# 音频处理函数
def adjust_length(waveform, target_length):
    if len(waveform) > target_length:
        return waveform[:target_length]
    elif len(waveform) < target_length:
        return np.pad(waveform, (0, target_length - len(waveform)), 'constant')
    else:
        return waveform


# 大模型分析器
# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()
# 初始化模型
interpreter = tf.lite.Interpreter(model_path="yamnet.tflite")
interpreter.allocate_tensors()
inputs = interpreter.get_input_details()
outputs = interpreter.get_output_details()
yamnet_classes = class_names('yamnet_class_map.csv')

# 音乐识别
key_proc = CNNKeyRecognitionProcessor()
tempo_proc = TempoEstimationProcessor(fps=100)
beat_proc = RNNBeatProcessor()
onset_peak_proc = OnsetPeakPickingProcessor(fps=100)
onset_proc = RNNOnsetProcessor()


# 大分析

def musicdetect(audio_path):
    # 调识别
    key = key_proc(audio_path)

    # 节奏估计
    beat_act = beat_proc(audio_path)
    tempo_estimates = tempo_proc(beat_act)

    # 起始检测
    onset_act = onset_proc(audio_path)
    onsets = onset_peak_proc(onset_act)

    # 创建 DataFrame
    data = {
        'Key': [key],
        'Tempo_Estimates': [tempo_estimates],
        'Onsets': [onsets]
    }
    df = pd.DataFrame(data)

    return df


def yamclass(file_name):
    # 读取和处理音频文件
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.SAMPLE_RATE:
        waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)

    # 模型期望的输入长度
    expected_input_length = 15600

    # 将音频数据分段
    num_segments = int(np.ceil(len(waveform) / expected_input_length))
    segments = [
        adjust_length(waveform[i * expected_input_length: (i + 1) * expected_input_length], expected_input_length) for i
        in range(num_segments)]

    # 随机选择10个片段
    random_segments = random.sample(segments, min(10, len(segments)))
    # 处理每个片段并记录预测结果
    predictions = []
    for segment in random_segments:
        interpreter.set_tensor(inputs[0]['index'], np.expand_dims(np.array(segment, dtype=np.float32), axis=0))
        interpreter.invoke()
        scores = interpreter.get_tensor(outputs[0]['index'])
        predictions.append(scores)

    # 汇总所有片段的预测结果，取每个类别的最大值
    predictions = np.vstack(predictions)
    final_prediction = np.max(predictions, axis=0)
    top10_i = np.argsort(final_prediction)[::-1][:10]

    # 将结果存储到DataFrame
    # 将结果存储到DataFrame
    final_df = pd.DataFrame({
        'file_name': [file_name],
        'class_1': yamnet_classes[top10_i[0]] if len(top10_i) > 0 else None,
        'confidence_voice_1': final_prediction[top10_i[0]] if len(top10_i) > 0 else None,
        'class_2': yamnet_classes[top10_i[1]] if len(top10_i) > 1 else None,
        'confidence_voice_2': final_prediction[top10_i[1]] if len(top10_i) > 1 else None,
        'class_3': yamnet_classes[top10_i[2]] if len(top10_i) > 2 else None,
        'confidence_voice_3': final_prediction[top10_i[2]] if len(top10_i) > 2 else None,
        'class_4': yamnet_classes[top10_i[3]] if len(top10_i) > 3 else None,
        'confidence_voice_4': final_prediction[top10_i[3]] if len(top10_i) > 3 else None,
        'class_5': yamnet_classes[top10_i[4]] if len(top10_i) > 4 else None,
        'confidence_voice_5': final_prediction[top10_i[4]] if len(top10_i) > 4 else None,
        'class_6': yamnet_classes[top10_i[5]] if len(top10_i) > 5 else None,
        'confidence_voice_6': final_prediction[top10_i[5]] if len(top10_i) > 5 else None,
        'class_7': yamnet_classes[top10_i[6]] if len(top10_i) > 6 else None,
        'confidence_voice_7': final_prediction[top10_i[6]] if len(top10_i) > 6 else None,
        'class_8': yamnet_classes[top10_i[7]] if len(top10_i) > 7 else None,
        'confidence_voice_8': final_prediction[top10_i[7]] if len(top10_i) > 7 else None,
        'class_9': yamnet_classes[top10_i[8]] if len(top10_i) > 8 else None,
        'confidence_voice_9': final_prediction[top10_i[8]] if len(top10_i) > 8 else None,
        'class_10': yamnet_classes[top10_i[9]] if len(top10_i) > 9 else None,
        'confidence_voice_10': final_prediction[top10_i[9]] if len(top10_i) > 9 else None,
    })
    final_df = final_df.drop(columns="file_name")
    return final_df


# def chromadetect(fullfilename):
#     # 加载音频文件
#     y, sr = librosa.load(fullfilename, sr=None)  # 默认加载为单声道，保持原始采样率
#
#     # 计算 Chroma 特征
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
#
#     # 按帧求和以获得整体强度
#     summed_chroma = np.sum(chroma, axis=1)  # 每个音高的总强度
#
#     # 找到强度最高的 5 个音高索引（降序排序）
#     top_5_indices = np.argsort(summed_chroma)[::-1][:5]  # 返回前 5 个索引
#
#     # 音高名称（C, C#, D, ..., B）
#     pitch_names = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
#     top_5_pitches = pitch_names[top_5_indices]
#
#     # 创建 DataFrame
#     data = {
#         'Pitch1': [top_5_pitches[0]],
#         'Pitch2': [top_5_pitches[1]],
#         'Pitch3': [top_5_pitches[2]],
#         'Pitch4': [top_5_pitches[3]],
#         'Pitch5': [top_5_pitches[4]],
#         'Chroma_Matrix': [chroma.tolist()]  # 将矩阵存为列表形式
#     }
#     df = pd.DataFrame(data)
#
#     return df


def birdclass(fullfilename, lat=None, lon=None):
    # 种类识别
    recording = Recording(analyzer, fullfilename, min_conf=0.1, lat=lat, lon=lon)
    recording.analyze()

    detections = recording.detections

    # 检查 detections 是否为空
    if not detections:
        final_df = pd.DataFrame({
            'file_name': [fullfilename],
            'bird_1': [None],
            'confidence_1': [None],
            'bird_2': [None],
            'confidence_2': [None],
            'bird_3': [None],
            'confidence_3': [None],
            'bird_4': [None],
            'confidence_4': [None],
            'bird_5': [None],
            'confidence_5': [None],
        })

    else:

        # Create a DataFrame for the detections
        dbird = pd.DataFrame(detections)
        # Group by 'common_name' and get the maximum confidence for each bird
        max_confidence = dbird.groupby('common_name')['confidence'].max().reset_index()
        # Sort by confidence in descending order
        max_confidence = max_confidence.sort_values(by='confidence', ascending=False).reset_index(drop=True)
        # Get the top 5 bird species
        top_5 = max_confidence.head(5)
        # 如果少于 5 种鸟类，用 NaN 填充
        top_5 = top_5.reindex(range(5)).reset_index(drop=True)

        # Create the final DataFrame with the required structure
        final_df = pd.DataFrame({
            'file_name': [fullfilename],
            'bird_1': top_5['common_name'].iloc[0] if len(top_5) > 0 else None,
            'confidence_1': top_5['confidence'].iloc[0] if len(top_5) > 0 else None,
            'bird_2': top_5['common_name'].iloc[1] if len(top_5) > 1 else None,
            'confidence_2': top_5['confidence'].iloc[1] if len(top_5) > 1 else None,
            'bird_3': top_5['common_name'].iloc[2] if len(top_5) > 2 else None,
            'confidence_3': top_5['confidence'].iloc[2] if len(top_5) > 2 else None,
            'bird_4': top_5['common_name'].iloc[3] if len(top_5) > 3 else None,
            'confidence_4': top_5['confidence'].iloc[3] if len(top_5) > 3 else None,
            'bird_5': top_5['common_name'].iloc[4] if len(top_5) > 4 else None,
            'confidence_5': top_5['confidence'].iloc[4] if len(top_5) > 4 else None,
        })
        final_df = final_df.drop(columns="file_name")
    return final_df


# 失真、广场舞的表情
def gen_label(df, df_label):
    for index, row in df.iterrows():
        try:

            # get the full filename of the corresponding row
            fullfilename = row['file']
            # Save file basename
            path, filename = os.path.split(fullfilename)
            row['file'] = filename
            row['filewithpath'] = fullfilename

            #### Load the original sound (16bits) and get the sampling frequency fs
            try:
                wave, fs = librosa.load(fullfilename, sr=None, mono=True)
                wave = wave - np.mean(wave)

            except:
                # Delete the row if the file does not exist or raise a value error (i.e. no EOF)
                df.drop(index, inplace=True)
                continue

            df_row = pd.DataFrame(row)
            df_row = df_row.T
            # if 'Date' in df_row.columns:
            #     df_row.rename(columns={'Date': 'Date_old'}, inplace=True)
            df_row.index.name = 'Date'
            # df_row = df_row.reset_index()
            try:
                df_row["isclipped"] = detect_clipping(fullfilename)
            except Exception as e:
                print(f"Error in detect_clipping for file {fullfilename}: {e}")
                df_row["isclipped"] = np.nan  # 如果出错，赋值为 NaN

            try:
                musicd = detect_music_and_speech(fullfilename)
                df_row["ismusic"] = musicd[0]
                df_row["isspeech"] = musicd[1]
            except Exception as e:
                print(f"Error in detect_music_and_speech for file {fullfilename}: {e}")
                df_row["ismusic"] = np.nan  # 如果出错，赋值为 NaN
                df_row["isspeech"] = np.nan

            try:
                dfclass = yamclass(fullfilename)
            except Exception as e:
                print(f"Error in yamclass for file {fullfilename}: {e}")
                dfclass = pd.DataFrame()  # 如果出错，赋值为空列表

            # add the row with scalar indices into the df_indices dataframe
            # create a row with the different scalar indices
            row_scalar_indices = pd.concat([df_row, dfclass], axis=1)
            df_label = pd.concat([df_label, row_scalar_indices])

        # # # Set back Date as index
        # df_label = df_label.set_index('Date')
        except Exception as e:
            # 捕获整个循环体内的异常，打印错误信息并跳过当前循环
            print(f"Error processing row {index} for file {fullfilename}: {e}")
            continue

    return df_label


# 失真、广场舞的表情
def gen_feature(df, df_indices, df_indices_per_bin, lat=None, lon=None, gain=42, vadc=2, sensitivity=-35):
    for index, row in df.iterrows():
        try:

            # get the full filename of the corresponding row
            fullfilename = row['file']
            # Save file basename
            path, filename = os.path.split(fullfilename)
            row['file'] = filename
            row['filewithpath'] = fullfilename

            #### Load the original sound (16bits) and get the sampling frequency fs
            try:
                wave, fs = librosa.load(fullfilename, sr=None, mono=True)
                wave = wave - np.mean(wave)

            except:
                # Delete the row if the file does not exist or raise a value error (i.e. no EOF)
                df.drop(index, inplace=True)
                continue

            """ =======================================================================
                            Computation in the time domain
            ========================================================================"""

            # Parameters of the audio recorder. This is not a mandatory but it allows
            # to compute the sound pressure level of the audio file (dB SPL) as a
            # sonometer would do.
            S = sensitivity  # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)
            G = gain  # Amplification gain (26dB (SM4 preamplifier))
            V = vadc

            # compute all the audio indices and store them into a DataFrame
            # dB_threshold and rejectDuration are used to select audio events.
            df_audio_ind = features.all_temporal_alpha_indices(
                s=wave,
                fs=fs,
                gain=G,
                vadc=V,
                sensibility=S,
                dB_threshold=3,
                rejectDuration=0.01,
                verbose=False,
                display=False,
            )

            """ =======================================================================
                            Computation in the frequency domain
            ========================================================================"""

            # Compute the Power Spectrogram Density (PSD) : Sxx_power
            Sxx_power, tn, fn, ext = sound.spectrogram(
                x=wave,
                fs=fs,
                window='hann',
                nperseg=1024,
                noverlap=1024 // 2,
                verbose=False,
                display=False,
                savefig=None
            )

            # compute all the spectral indices and store them into a DataFrame
            # flim_low, flim_mid, flim_hi corresponds to the frequency limits in Hz
            # that are required to compute somes indices (i.e. NDSI)
            # if R_compatible is set to 'soundecology', then the output are similar to
            # soundecology R package.
            # mask_param1 and mask_param2 are two parameters to find the regions of
            # interest (ROIs). These parameters need to be adapted to the dataset in
            # order to select ROIs
            try:
                df_spec_ind, df_spec_ind_per_bin = features.all_spectral_alpha_indices(
                    Sxx_power=Sxx_power,
                    tn=tn,
                    fn=fn,
                    flim_low=[0, 1500],
                    flim_mid=[1500, 8000],
                    flim_hi=[8000, 20000],
                    gain=G,
                    sensitivity=S,
                    vadc=V,
                    verbose=False,
                    R_compatible='soundecology',
                    mask_param1=6,
                    mask_param2=0.5,
                    display=False)
            except Exception as e:
                # 如果出错，则返回空的 df_spec_ind, df_spec_ind_per_bin
                print(f"Error in all_spectral_alpha_indices: {e}")
                df_spec_ind = pd.DataFrame()
                df_spec_ind_per_bin = pd.DataFrame()

            """ =======================================================================
                            Create a dataframe
            ========================================================================"""
            # First, we create a dataframe from row that contains the date and the
            # full filename. This is done by creating a DataFrame from row (ie. TimeSeries)
            # then transposing the DataFrame.
            df_row = pd.DataFrame(row)
            df_row = df_row.T
            # df_row.index.name = 'Date'
            df_row = df_row.reset_index(drop=True)

            lat = row['lat'] if pd.notna(row['lat']) else None
            lon = row['lng'] if pd.notna(row['lng']) else None

            try:
                final_df = birdclass(fullfilename, lat=lat, lon=lon)
            except Exception as e:
                print(f"Error in processing: {str(e)}")
                final_df =  pd.DataFrame()
            try:
                dfmu = musicdetect(fullfilename)
            except Exception as e:
                print(f"Error in processing: {str(e)}")
                dfmu =pd.DataFrame()

            try:
                chord_HMM, chord_max, X_chroma, Fs_X = chordrecog(fullfilename)
                dfchro = process_single_sample(chord_HMM, X_chroma, chord_max, chord_labels, Fs_X)
            except Exception as e:
                print(f"Error in processing: {str(e)}")
                dfchro = pd.DataFrame()

            # create a row with the different scalar indices
            row_scalar_indices = pd.concat(
                [df_row, df_audio_ind, df_spec_ind, final_df, dfmu, dfchro],
                axis=1
            )
            # add the row with scalar indices into the df_indices dataframe
            df_indices = pd.concat([df_indices, row_scalar_indices])

            # create a row with the different vector indices
            row_vector_indices = pd.concat(
                [df_row, df_spec_ind_per_bin],
                axis=1)

            # add vector indices into the df_indices_per_bin dataframe
            df_indices_per_bin = pd.concat([df_indices_per_bin, row_vector_indices])

        # # # Set back Date as index
        # df_indices = df_indices.set_index('Date')
        # df_indices_per_bin = df_indices_per_bin.set_index('Date')

        except Exception as e:
            # Log the error and continue to the next row
            print(f"Error processing row {index}: {e}")
            continue

    return df_indices, df_indices_per_bin


# 读取与处理


# 遍历 output_path 下的所有子文件夹并解析数据
def collect_data(intput_path):
    all_dfs = []

    # 遍历所有文件夹
    for root, dirs, files in os.walk(intput_path):
        for dir_name in dirs:
            gen_path = os.path.join(root, dir_name)
            os.makedirs(gen_path, exist_ok=True)

            # 调用 date_parser 解析数据
            try:
                df = date_parser(gen_path, dateformat='', verbose=False, extension='.mp3')
                all_dfs.append(df.reset_index(drop=True))
            except Exception as e:
                print(f"Error parsing data in {gen_path}: {e}")

    # 将所有 DataFrame 纵向拼接
    if all_dfs:
        concatenated_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        return concatenated_df
    else:
        print("No data found in the specified directories.")
        return None


def extract_bird_details(file_path):
    # 提取鸟的完整名称和种名
    try:
        # 标准化路径并获取倒数第二层文件夹名称
        normalized_path = os.path.normpath(file_path)
        bird_full_name = os.path.split(os.path.split(normalized_path)[0])[1]

        # 使用 '_' 分割完整名称
        parts = bird_full_name.split('_')
        return parts
    except IndexError:
        return None, None  # 若解析失败，返回空值


def extract_folder(file_path):
    # 提取鸟的完整名称和种名
    try:
        # 标准化路径并获取倒数第二层文件夹名称
        normalized_path = os.path.normpath(file_path)
        name = os.path.split(os.path.split(normalized_path)[0])[1]

        return name
    except IndexError:
        return None, None  # 若解析失败，返回空值


def clean_time(time):
    # 替换非法字符并标准化时间格式
    if '?' in time or time.lower() in ['xx:xx', '?:?', '??', 'night', 'early', 'morning', 'afternoon', 'am']:
        return '12:00'
    time = time.replace('.', ':').replace('h', '').strip()
    if '-' in time:
        time = time.replace('-', ':')
    try:
        # 处理不同格式的时间
        if len(time.split(':')) == 3:  # 处理带秒的时间
            parsed_time = pd.to_datetime(time, format='%H:%M:%S', errors='coerce')
        elif len(time.split(':')) == 2:  # 处理小时:分钟格式
            parsed_time = pd.to_datetime(time, format='%H:%M', errors='coerce')
        elif time.isdigit() and len(time) == 4:  # 处理1100这种格式
            parsed_time = pd.to_datetime(time[:2] + ':' + time[2:], format='%H:%M', errors='coerce')
        else:
            return '12:00'

        if pd.notnull(parsed_time):
            return parsed_time.strftime('%H:%M')
        else:
            return '12:00'
    except ValueError:
        return '12:00'


# 清理日期函数
def clean_date(date_str):
    if pd.isnull(date_str) or not isinstance(date_str, str):
        return '2008-01-01'  # 空值或非字符串直接返回默认日期

    parts = date_str.split('-')
    if len(parts) != 3:
        return '2008-01-01'  # 非法格式，返回默认日期

    year, month, day = parts

    # 处理年份
    if not year.isdigit() or int(year) < 1900 or int(year) > 2100:
        return '2008-01-01'  # 年份不合法，返回默认日期

    # 处理月份
    if month == '00' or not month.isdigit() or int(month) < 1 or int(month) > 12:
        month = '01'  # 默认替换为 1 月

    # 处理日期
    if day == '00' or not day.isdigit() or int(day) < 1 or int(day) > 31:
        day = '01'  # 默认替换为 1 日

    cleaned_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return cleaned_date


# 将 df_dataset 的 'date' 和 'time' 合并为新的 'Date' 列
def combine_date_time(df):
    # 应用时间清理函数
    df['time'] = df['time'].apply(clean_time)
    df['date'] = df['date'].apply(clean_date)
    # 合并日期和时间
    df['Date'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M', errors='coerce')
    if df['Date'].isnull().any():
        print("Warning: Some date-time combinations could not be parsed")
    return df


# 加速的 lambda 函数
@numba.jit(nopython=True)
def extract_real_filename(filename):
    parts = filename.split('/')[-1].split('\\')[-1]
    return parts.split('.')[0]  # 提取文件名核心部分


def extract_bird_details(file_path):
    # 提取鸟的完整名称和种名
    try:
        # 标准化路径并获取倒数第二层文件夹名称
        normalized_path = os.path.normpath(file_path)
        bird_full_name = os.path.split(os.path.split(normalized_path)[0])[1]

        # 使用 '_' 分割完整名称
        parts = bird_full_name.split('_')
        return parts
    except IndexError:
        return None, None  # 若解析失败，返回空值


# %%
# ========== 断点管理函数 ==========
def load_progress(progress_file):
    """Load or initialize progress from a JSON file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            return json.load(file)
    else:
        return {'origin': 0, 'spe': 0, 'sub': 0}  # 默认结构，可按需调整


def save_progress(progress_file, progress_data):
    """Save the current progress to a JSON file."""
    with open(progress_file, 'w') as file:
        json.dump(progress_data, file)


def merge_and_label(df_indices, df_indices_per_bin, df_label, process_label):
    """
    同时支持按 file 或 filewithpath 进行合并（OR 逻辑），
    最终只保留左侧(即 df_indices / df_indices_per_bin 中的)同名列，
    并添加一个标记列 'process'。

    合并流程：
    1) 先按 'file' 进行合并 -> merged_file
    2) 再按 'filewithpath' 进行合并 -> merged_path
    3) 将 merged_file 和 merged_path 竖向拼接 (concat)
    4) 根据需要的列去重，最终加上 'process' 列。
    """

    # --- 1) 按 'file' 合并 ---
    merged_file = pd.merge(
        df_indices,
        df_indices_per_bin,
        on='file',
        how='left',
        suffixes=('', '_r1')  # 避免列名冲突
    )
    merged_file = pd.merge(
        merged_file,
        df_label,
        on='file',
        how='left',
        suffixes=('', '_r2')
    )

    # --- 2) 按 'filewithpath' 合并 ---
    merged_path = pd.merge(
        df_indices,
        df_indices_per_bin,
        on='filewithpath',
        how='left',
        suffixes=('', '_r3')
    )
    merged_path = pd.merge(
        merged_path,
        df_label,
        on='filewithpath',
        how='left',
        suffixes=('', '_r4')
    )

    # --- 3) 竖向拼接，去重 ---
    merged_df = pd.concat([merged_file, merged_path], ignore_index=True)
    merged_df.drop_duplicates(subset=['file', 'filewithpath'], keep='first', inplace=True)
    # 1) 定义一个后缀列表，你实际用到了哪些后缀就写哪些：
    suffix_list = ['_r1', '_r2', '_r3', '_r4']

    # 2) 根据后缀批量找出要删除的列
    cols_to_drop = []
    for suffix in suffix_list:
        # 找出所有以 suffix 结尾的列
        cols = [col for col in merged_df.columns if col.endswith(suffix)]
        cols_to_drop.extend(cols)

    # 3) 一次性删除这些列
    merged_df.drop(columns=cols_to_drop, inplace=True)

    # --- 4) 添加 'process' 标记列 ---
    merged_df["process"] = process_label

    return merged_df


def load_progress(progress_file):
    """Load or initialize progress from a JSON file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            return json.load(file)
    else:
        return {'origin': 0, 'spe': 0, 'sub': 0}  # 默认结构，可按需调整


def save_progress(progress_file, progress_data):
    """Save the current progress to a JSON file."""
    with open(progress_file, 'w') as file:
        json.dump(progress_data, file)


def append_to_csv(dataframe, csv_file):
    """Append a dataframe to a CSV file."""
    if not os.path.isfile(csv_file):
        dataframe.to_csv(csv_file, index=False)
    else:
        dataframe.to_csv(csv_file, mode='a', header=False, index=False)


def append_to_pickle(dataframe, pkl_file):
    """Append a dataframe to a pickle file."""
    if not os.path.isfile(pkl_file):
        dataframe.to_pickle(pkl_file)
    else:
        existing_df = pd.read_pickle(pkl_file)
        combined_df = pd.concat([existing_df, dataframe], ignore_index=True)
        combined_df.to_pickle(pkl_file)


def process_and_save(lat=None, lon=None, gain=42, vadc=2, sensitivity=-35, XC_ROOTDIR=None, XC_DIR=None):
    datapath = XC_ROOTDIR
    os.makedirs(datapath, exist_ok=True)
    intput_path = os.path.join(datapath, XC_DIR)
    os.makedirs(intput_path, exist_ok=True)
    log_file_path = os.path.join('process_log.txt')
    with open(log_file_path, 'w') as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(
            log_file):
        start_time = time.time()

        df_dataset = pd.read_pickle('df_dataset.pkl')
        # 调用函数
        df = collect_data(intput_path)

        if df is not None and 'file' in df.columns:
            # 提取文件名核心部分
            df['realfilename'] = df['file'].apply(extract_real_filename)
            df['foldername'] = df['file'].apply(extract_folder)

            # 提取鸟的完整名称和种名
            bird_details = df['file'].apply(lambda x: extract_bird_details(x) if isinstance(x, str) else (None, None))
            df['birdenname'] = bird_details.map(lambda x: x[1])
            df['birdspgen'] = bird_details.map(lambda x: x[0])

        df_dataset = combine_date_time(df_dataset)
        filename_to_date_map = df_dataset.set_index('realfilename')['Date'].to_dict()
        default_date = '2008-01-01'  # 设置默认日期
        df['Date'] = df['realfilename'].map(filename_to_date_map).fillna(default_date)
        df.index = df.Date
        merged_df = pd.merge(df, df_dataset, on='realfilename', how='left', suffixes=('', '_dataset'))

        # ---- 2) 读取进度文件(当前目录下) ----
        progress_file = os.path.join(os.getcwd(), 'progress.json')
        progress = load_progress(progress_file)

        # 若已存在 "folders" 字段也行，否则初始化
        if "folders" not in progress:
            # 这里仅是辅助记录，如果你还需要 'origin','spe','sub' 保持原有逻辑，也可保留
            progress["folders"] = {}

            # ---- 3) 按 foldername 分组并批量处理 ----
        if 'foldername' not in merged_df.columns:
            print(
                "Warning: there is no foldername column in merged_df, make sure that collect_data or other logic has generated the column.")
            return

        grouped = merged_df.groupby('foldername')

        # 准备批处理参数
        batch_size = 90
        merge_interval = 3
        all_batches = []  # 用于每处理 merge_interval 个批次，就合并一次

        # ---- 4) 针对每个文件夹断点续跑 ----
        for folder_name, group_df in grouped:
            # 关键一步！！！！
            group_df = group_df.reset_index(drop=True)

            # 该文件夹上次处理到的批次数(默认0)
            start_batch_for_folder = progress["folders"].get(folder_name, 0)

            num_files = len(group_df)
            # 从上次处理过的批次继续
            for i in range(start_batch_for_folder * batch_size, num_files, batch_size):
                batch_end = min(i + batch_size, num_files)
                current_batch = group_df.iloc[i:batch_end].copy()


                # ------ 生成标签 ------
                df_label = gen_label(current_batch.copy(), pd.DataFrame())

                # ------ 生成特征 ------
                df_indices, df_indices_per_bin = pd.DataFrame(), pd.DataFrame()
                df_indices, df_indices_per_bin = gen_feature(
                    current_batch.copy(),
                    df_indices,
                    df_indices_per_bin,
                    lat=lat, lon=lon, gain=gain, vadc=vadc, sensitivity=sensitivity
                )

                # ------ 合并特征和标签 ------
                merged_batch_df = merge_and_label(df_indices, df_indices_per_bin, df_label, process_label="origin")

                # 暂存进 all_batches
                all_batches.append(merged_batch_df)


                # ------ 保存当前批次文件(防止中断丢失) ------
                label_path = os.path.join(datapath, "origindata")
                os.makedirs(label_path, exist_ok=True)

                batch_file_suffix = f"{folder_name}_batch_{i // batch_size + 1}"
                merged_batch_df.to_csv(os.path.join(label_path, f"{batch_file_suffix}.csv"), index=False)
                merged_batch_df.to_pickle(os.path.join(label_path, f"{batch_file_suffix}.pkl"))

                # ------ 每处理若干批次就合并并追加保存 ------
                if len(all_batches) >= merge_interval:
                    final_merged_df = pd.concat(all_batches, ignore_index=True)
                    append_to_csv(final_merged_df, os.path.join(datapath, "merge_df.csv"))
                    append_to_pickle(final_merged_df, os.path.join(datapath, "merge_df.pkl"))
                    all_batches = []

                # ------ 更新进度并保存 ------
                progress["folders"][folder_name] = i // batch_size + 1
                save_progress(progress_file, progress)

        # 若还有剩余没合并完的
        if all_batches:
            final_merged_df = pd.concat(all_batches, ignore_index=True)
            append_to_csv(final_merged_df, os.path.join(datapath, "merge_df.csv"))
            append_to_pickle(final_merged_df, os.path.join(datapath, "merge_df.pkl"))
            all_batches = []

        # ---- 5) 最终大合并：扫描 origindata 下所有批文件再合并 ----
        all_merged_dfs = []
        label_path = os.path.join(datapath, "origindata")
        for root, _, files in os.walk(label_path):
            for file in files:
                if file.endswith('.pkl'):
                    temp_df = pd.read_pickle(os.path.join(root, file))
                    all_merged_dfs.append(temp_df)

        if all_merged_dfs:
            final_merged_df_all = pd.concat(all_merged_dfs, ignore_index=True)
            final_merged_df_all.to_csv(os.path.join(datapath, "final_merge_df_all_folders.csv"), index=False)
            final_merged_df_all.to_pickle(os.path.join(datapath, "final_merge_df_all_folders.pkl"))

        end_time = time.time()
        print(f"Processing complete, total time: {end_time - start_time:.2f} seconds")


# %%
XC_ROOTDIR = './data'
XC_DIR = 'global_dataset'
if __name__ == "__main__":
    process_and_save(lat=None, lon=None, gain=42, vadc=2, sensitivity=-35, XC_ROOTDIR=XC_ROOTDIR, XC_DIR=XC_DIR)





