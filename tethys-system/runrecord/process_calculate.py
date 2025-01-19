import librosa
import librosa.feature
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



#%%
#禁止提醒
import warnings
from scipy.io.wavfile import WavFileWarning
from sklearn.exceptions import InconsistentVersionWarning
 # Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=WavFileWarning)
import contextlib

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
                                                                            version='STFT', frepass=False, ifroi=ifroi)
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





#%%
SPECTRAL_FEATURES=['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf',
    'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS','EPS_KURT','EPS_SKEW','ACI',
    'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
    'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
    'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
    'AGI','ROItotal','ROIcover'
    ]

TEMPORAL_FEATURES=['ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
    'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount',
    'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount'
    ]
#%%
#基础计算函数
#失真检验
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



#可恶的广场舞
def detect_music_and_speech(audio_file):
    # 加载音频文件
    signal, sr = librosa.load(audio_file, sr=None)  # 使用音频本身的采样率
    signal=signal-np.mean(signal)

    # 使用pyAudioAnalysis进行音乐和语音检测
    result_index, probabilities, classes = aT.file_classification(audio_file, "pretrained_models/data/models/svm_rbf_sm", "svm")

    # 获取最可能的类别索引
    result_index = int(result_index)  # 确保索引是整数

    # 解析返回结果
    is_music = classes[result_index] == 'music'
    is_speech = classes[result_index] == 'speech'
    music_probability = probabilities[classes.index('music')]
    speech_probability = probabilities[classes.index('speech')]

    return is_music,is_speech

# 读取类名
def class_names(class_map_csv):
    """Read the class name definition file and return a list of strings."""
    with open(class_map_csv) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)   # Skip header
        return np.array([display_name for (_, _, display_name) in reader])

# 音频处理函数
def adjust_length(waveform, target_length):
    if len(waveform) > target_length:
        return waveform[:target_length]
    elif len(waveform) < target_length:
        return np.pad(waveform, (0, target_length - len(waveform)), 'constant')
    else:
        return waveform



#%%

#大模型分析器
# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()
# 初始化模型
interpreter = tf.lite.Interpreter(model_path="yamnet.tflite")
interpreter.allocate_tensors()
inputs = interpreter.get_input_details()
outputs = interpreter.get_output_details()
yamnet_classes = class_names('yamnet_class_map.csv')

#音乐识别
key_proc = CNNKeyRecognitionProcessor()
tempo_proc = TempoEstimationProcessor(fps=100)
beat_proc = RNNBeatProcessor()
onset_peak_proc = OnsetPeakPickingProcessor(fps=100)
onset_proc = RNNOnsetProcessor()
#%%
#大分析

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
    segments = [adjust_length(waveform[i * expected_input_length: (i + 1) * expected_input_length], expected_input_length) for i in range(num_segments)]

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
    final_df=final_df.drop(columns="file_name")
    return final_df

def birdclass(fullfilename, lat=None,lon=None):

    #种类识别
    recording = Recording(analyzer,fullfilename,min_conf=0.1, lat=lat,lon=lon)
    recording.analyze()

    detections=recording.detections

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
        final_df=final_df.drop(columns="file_name")
    return final_df



#失真、广场舞的表情
def gen_label(df,df_label):
    for index, row in df.iterrows() :

        # get the full filename of the corresponding row
        fullfilename = row['file']
        # Save file basename
        path, filename = os.path.split(fullfilename)
        row['file']=filename

        #### Load the original sound (16bits) and get the sampling frequency fs
        try :
            wave,fs = librosa.load(fullfilename,sr=None,mono=True)
            wave=wave-np.mean(wave)

        except:
            # Delete the row if the file does not exist or raise a value error (i.e. no EOF)
            df.drop(index, inplace=True)
            continue

        df_row = pd.DataFrame(row)
        df_row =df_row.T
        df_row.index.name = 'Date'
        df_row = df_row.reset_index()
        df_row["isclipped"]=detect_clipping(fullfilename)
        musicd=detect_music_and_speech(fullfilename)
        df_row["ismusic"]=musicd[0]
        df_row["isspeech"]=musicd[1]
        dfclass=yamclass(fullfilename)

        # add the row with scalar indices into the df_indices dataframe
         # create a row with the different scalar indices
        row_scalar_indices =pd.concat([df_row, dfclass],axis=1)
        df_label = pd.concat([df_label, row_scalar_indices])

    # # # Set back Date as index
    # df_label = df_label.set_index('Date')

    return df_label


#失真、广场舞的表情
def gen_feature(df,df_indices,df_indices_per_bin, lat=None,lon=None,gain=42,vadc=2,sensitivity=-35):
    for index, row in df.iterrows() :

        # get the full filename of the corresponding row
        fullfilename = row['file']
        # Save file basename
        path, filename = os.path.split(fullfilename)
        row['file']=filename

        #### Load the original sound (16bits) and get the sampling frequency fs
        try :
            wave,fs = librosa.load(fullfilename,sr=None,mono=True)
            wave=wave-np.mean(wave)

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
        S = sensitivity         # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)
        G = gain       # Amplification gain (26dB (SM4 preamplifier))
        V=vadc

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
        Sxx_power,tn,fn,ext = sound.spectrogram (
            x=wave,
            fs=fs,
            window='hann',
            nperseg=1024,
            noverlap=1024//2,
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
        df_spec_ind, df_spec_ind_per_bin = features.all_spectral_alpha_indices(
            Sxx_power=Sxx_power,
            tn=tn,
            fn=fn,
            flim_low=[0,1500],
            flim_mid=[1500,8000],
            flim_hi=[8000,20000],
            gain=G,
            sensitivity=S,
            vadc=V,
            verbose=False,
            R_compatible='soundecology',
            mask_param1=6,
            mask_param2=0.5,
            display=False)

        """ =======================================================================
                        Create a dataframe
        ========================================================================"""
        # First, we create a dataframe from row that contains the date and the
        # full filename. This is done by creating a DataFrame from row (ie. TimeSeries)
        # then transposing the DataFrame.
        df_row = pd.DataFrame(row)
        df_row =df_row.T
        df_row.index.name = 'Date'
        df_row = df_row.reset_index()

        final_df=birdclass(fullfilename, lat=lat,lon=lon)
        dfmu=musicdetect(fullfilename)
        chord_HMM, chord_max, X_chroma, Fs_X = chordrecog(fullfilename)
        dfchro = process_single_sample(chord_HMM, X_chroma, chord_max, chord_labels, Fs_X)



        # create a row with the different scalar indices
        row_scalar_indices =pd.concat(
            [df_row, df_audio_ind, df_spec_ind,final_df,dfmu,dfchro],
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



    return df_indices,df_indices_per_bin
#%%
#数据后处理
# 按 'file' 和 'Date' 进行合并，并去除重复的列
def merge_and_label(df_indices, df_indices_per_bin, df_label, process_label):
    merged_df = pd.merge(df_indices, df_indices_per_bin, on=['Date'], how='outer')
    merged_df = pd.merge(merged_df, df_label, on=['Date'], how='outer')
    merged_df["process"] = process_label
    return merged_df


def load_progress(progress_file):
    """Load or initialize progress from a JSON file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            return json.load(file)
    else:
        return {'origin': 0, 'spe': 0, 'sub': 0}

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



def process_and_save(datapath, lat=None, lon=None, gain=42, vadc=2, sensitivity=-35):
    log_file_path = os.path.join(datapath, 'process_log.txt')
    with open(log_file_path, 'w') as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        start_time = time.time()
        progress_file = os.path.join(datapath, 'progress.json')
        progress = load_progress(progress_file)

        # Create directories
        os.makedirs(datapath, exist_ok=True)
        output_path_ehyspe = os.path.join(datapath, "ehyspe")
        output_path_ehysub = os.path.join(datapath, "ehysub")
        output_path_origin = os.path.join(datapath, "origin")
        os.makedirs(output_path_ehyspe, exist_ok=True)
        os.makedirs(output_path_ehysub, exist_ok=True)
        os.makedirs(output_path_origin, exist_ok=True)

        # Define batch size and merge interval
        batch_size = 150
        merge_interval = 5  # Merge every 5 batches

        # Load and parse data
        df = date_parser(output_path_origin, dateformat='%Y%m%d_%H%M%S', verbose=False)
        df2 = date_parser(output_path_ehyspe, dateformat='%Y%m%d_%H%M%S', verbose=False)
        df3 = date_parser(output_path_ehysub, dateformat='%Y%m%d_%H%M%S', verbose=False)

        # Initialize storage for merged data
        all_batches = []

        # Process each dataset
        datasets = [
            (df, output_path_origin, 'origin', progress['origin']),
            (df2, output_path_ehyspe, 'spe', progress['spe']),
            (df3, output_path_ehysub, 'sub', progress['sub'])
        ]

        for df_batch, path, label, start_batch in datasets:

            if label != 'origin':
                continue  # Skip processing for any label other than 'origin'



            label_path = os.path.join(datapath, label + "data")
            dflabelpath = os.path.join(datapath, "labeldata")
            os.makedirs(dflabelpath, exist_ok=True)
            os.makedirs(label_path, exist_ok=True)

            num_files = len(df_batch)
            for i in range(start_batch * batch_size, num_files, batch_size):
                batch_end = min(i + batch_size, num_files)  # Ensure we don't go out of bounds
                if label == 'origin':
                    # Generate labels using current batch of df
                    df_label = gen_label(df_batch[i:batch_end].copy(), pd.DataFrame())
                    # Save origin labels for future use
                    df_label.to_pickle(os.path.join(dflabelpath, f"origin_label_batch_{i // batch_size + 1}.pkl"))
                else:
                    # Load corresponding df_label batch from origin
                    df_label_path = os.path.join(datapath, "labeldata", f"origin_label_batch_{i // batch_size + 1}.pkl")
                    if os.path.exists(df_label_path):
                        df_label = pd.read_pickle(df_label_path)
                    else:
                        raise FileNotFoundError(f"Label file not found: {df_label_path}")

                # Process current batch with the corresponding df_label
                df_indices, df_indices_per_bin = pd.DataFrame(), pd.DataFrame()
                df_indices, df_indices_per_bin = gen_feature(df_batch[i:batch_end].copy(), df_indices, df_indices_per_bin, lat=lat, lon=lon, gain=gain, vadc=vadc, sensitivity=sensitivity)
                merged_df = merge_and_label(df_indices, df_indices_per_bin, df_label, label)
                all_batches.append(merged_df)

                # Save each batch's output uniquely
                batch_file_suffix = f"{label}_batch_{i // batch_size + 1}"
                merged_df.to_csv(os.path.join(label_path, f"{batch_file_suffix}.csv"), index=False)
                merged_df.to_pickle(os.path.join(label_path, f"{batch_file_suffix}.pkl"))

                # Every 10 batches, merge and save
                if len(all_batches) >= merge_interval:
                    final_merged_df = pd.concat(all_batches, ignore_index=True)
                    append_to_csv(final_merged_df, os.path.join(datapath, f"merge_df.csv"))
                    append_to_pickle(final_merged_df, os.path.join(datapath, f"merge_df.pkl"))
                    all_batches = []  # Reset for next merge

                # Update progress and save
                progress[label] = i // batch_size + 1
                save_progress(progress_file, progress)

        # Update progress for 'spe' and 'sub' to match 'origin'
        progress['spe'] = progress['origin']
        progress['sub'] = progress['origin']
        save_progress(progress_file, progress)


        # Merge all batches from batch directories
        all_merged_dfs = []
        for label in ['origin', 'spe', 'sub']:
            label_path = os.path.join(datapath, label + "data")
            for root, _, files in os.walk(label_path):
                for file in files:
                    if file.endswith('.pkl'):
                        df = pd.read_pickle(os.path.join(root, file))
                        all_merged_dfs.append(df)

        if all_merged_dfs:
            final_merged_df = pd.concat(all_merged_dfs, ignore_index=True)
            final_merged_df.to_csv(os.path.join(datapath, "final_merge_df_all_batches.csv"), index=False)
            final_merged_df.to_pickle(os.path.join(datapath, "final_merge_df_all_batches.pkl"))

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Calculation processing time: {total_time:.2f} seconds")



#%%
# 读取配置文件
with open('config.json', 'r') as file:
    config = json.load(file)
# 从配置文件获取参数
datapath = config['datapath']
lat = config['lat']
lon = config['lon']
gain = config['gain']
Vadc = config['Vadc']
sensitivity = config['sensitivity']
# 使用配置文件中的参数调用函数
process_and_save(datapath, lat=lat, lon=lon, gain=gain, vadc=Vadc, sensitivity=sensitivity)