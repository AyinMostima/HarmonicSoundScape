import os
import pandas as pd
import json

# 设定目录
base_dir = "/lustre/home/acct-yinshan/yinshan"
target_folder = "voiceprocess2024"
output_folder = os.path.join(base_dir, "voicesheetoutput")

# 创建输出目录（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 找到目标文件夹
target_path = os.path.join(base_dir, target_folder)

# 列出目标文件夹下的所有文件夹
subfolders = [f.name for f in os.scandir(target_path) if f.is_dir()]

# 读取检查点文件
checkpoint_file = os.path.join(output_folder, "finalmergedone.json")
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        processed_folders = json.load(f)
else:
    processed_folders = []

print("Found subfolders:", subfolders)

# 逐个处理文件夹
for folder in subfolders:
    if folder in processed_folders:
        print(f"Skipping folder {folder}, already processed.")
        continue

    selected_path = os.path.join(target_path, folder)
    data_folder = os.path.join(selected_path, "data")
    progress_file = os.path.join(data_folder, "progress.json")

    try:
        place = ''.join(filter(str.isalpha, folder.split('-')[0]))
        level = int(folder.split('-')[-1])
    except ValueError:
        print(f"Skipping folder {folder}, invalid format.")
        continue

    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)

        if progress_data["origin"] == progress_data["spe"] == progress_data["sub"]:
            pkl_file = os.path.join(data_folder, "final_merge_df_all_batches.pkl")
            if os.path.exists(pkl_file):
                df = pd.read_pickle(pkl_file)
                df['place'] = place
                df['level'] = level
                output_file = os.path.join(output_folder, f"merge_df_{folder}.pkl")
                df.to_pickle(output_file)
                print(f"Added data from {pkl_file} in folder {folder}")

                # 更新检查点文件
                processed_folders.append(folder)
                with open(checkpoint_file, 'w') as f:
                    json.dump(processed_folders, f)
            else:
                print(f"{pkl_file} does not exist in folder {folder}.")
        else:
            print(f"origin, spe, and sub are not equal in folder {folder}. No action needed.")
    else:
        print(f"Progress file {progress_file} does not exist in folder {folder}.")
