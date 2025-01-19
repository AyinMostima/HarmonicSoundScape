import os
import pandas as pd
import pickle
import json

# 设定目录
base_dir = "/lustre/home/acct-yinshan/yinshan"
target_folder = os.path.join(base_dir, "voicesheetoutput")
output_folder = os.path.join(target_folder, "variabletype")
progress_file = os.path.join(output_folder, "variabletypeprocess.json")
os.makedirs(output_folder, exist_ok=True)

# 定义已知的特征集
KNOWN_FEATURES = {
    'spectral_feature': ['MEANf', 'VARf', 'SKEWf', 'KURTf', 'NBPEAKS', 'LEQf', 'ENRf', 'BGNf', 'SNRf', 'Hf', 'EAS', 'ECU', 'ECV', 'EPS', 'EPS_KURT', 'EPS_SKEW', 'ACI', 'NDSI', 'rBA', 'AnthroEnergy', 'BioEnergy', 'BI', 'ROU', 'ADI', 'AEI', 'LFC', 'MFC', 'HFC', 'ACTspFract', 'ACTspCount', 'ACTspMean', 'EVNspFract', 'EVNspMean', 'EVNspCount', 'TFSD', 'H_Havrda', 'H_Renyi', 'H_pairedShannon', 'H_gamma', 'H_GiniSimpson', 'RAOQ', 'AGI', 'ROItotal', 'ROIcover'],
    'temporal_feature': ['ZCR', 'MEANt', 'VARt', 'SKEWt', 'KURTt', 'LEQt', 'BGNt', 'SNRt', 'MED', 'Ht', 'ACTtFraction', 'ACTtCount', 'ACTtMean', 'EVNtFraction', 'EVNtMean', 'EVNtCount'],
    'bin_feature': ['MEANt_per_bin', 'VARt_per_bin', 'SKEWt_per_bin', 'KURTt_per_bin', 'LEQf_per_bin', 'ENRf_per_bin', 'BGNf_per_bin', 'SNRf_per_bin', 'Ht_per_bin', 'ACI_per_bin', 'ROU_per_bin', 'ACTspFract_per_bin', 'ACTspCount_per_bin', 'EVNspFract_per_bin', 'EVNspMean_per_bin', 'EVNspCount_per_bin', 'AGI_per_bin']
}
BASE_VARS = ['Date', 'process', 'place', 'level', 'file_x']
ALL_FEATURES = set(sum(KNOWN_FEATURES.values(), []))

# 读取或初始化进度文件
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress_data = json.load(f)
else:
    progress_data = {}

def update_progress(filename, start_date, end_date):
    if filename not in progress_data:
        progress_data[filename] = {"processed": True, "start_date": start_date, "end_date": end_date}
    else:
        progress_data[filename]["start_date"] = min(progress_data[filename]["start_date"], start_date)
        progress_data[filename]["end_date"] = max(progress_data[filename]["end_date"], end_date)
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

def process_files():
    for file in sorted(os.listdir(target_folder)):
        if file.endswith('.pkl') and file not in progress_data:
            file_path = os.path.join(target_folder, file)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            data['Date'] = pd.to_datetime(data['Date'])
            start_date = data['Date'].min().strftime('%Y%m%d')
            end_date = data['Date'].max().strftime('%Y%m%d')

            for name, features in KNOWN_FEATURES.items():
                final_csv_path = os.path.join(output_folder, f'{name}.csv')
                selected_data = data[BASE_VARS + features]
                selected_data.to_csv(final_csv_path, mode='a', header=not os.path.exists(final_csv_path), index=False)

            # Identify and process 'musiclearning' features
            data_columns = set(data.columns) - ALL_FEATURES - set(BASE_VARS)
            if data_columns:  # If there are any other columns
                final_csv_path = os.path.join(output_folder, f'musiclearning.csv')
                selected_data = data[BASE_VARS + list(data_columns)]
                selected_data.to_csv(final_csv_path, mode='a', header=not os.path.exists(final_csv_path), index=False)

            update_progress(file, start_date, end_date)

    # Rename files to include date ranges
    rename_files()

def rename_files():
    for feature_set in list(KNOWN_FEATURES.keys()) + ['musiclearning']:
        file_path = os.path.join(output_folder, f'{feature_set}.csv')
        if os.path.exists(file_path):
            dates = aggregate_dates(progress_data)
            new_file_path = os.path.join(output_folder, f'{feature_set}_{dates["start_date"]}_{dates["end_date"]}.csv')
            os.rename(file_path, new_file_path)

def aggregate_dates(progress_data):
    min_date = min(entry["start_date"] for entry in progress_data.values() if "start_date" in entry)
    max_date = max(entry["end_date"] for entry in progress_data.values() if "end_date" in entry)
    return {"start_date": min_date, "end_date": max_date}

process_files()
