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

print("Found subfolders:", subfolders)

all_dfs = []

# 遍历所有子文件夹
for folder in subfolders:
    selected_path = os.path.join(target_path, folder)

    # 查看data文件夹中的progress.json文件
    data_folder = os.path.join(selected_path, "data")
    progress_file = os.path.join(data_folder, "progress.json")

    # 获取文件夹名中的place和level
    try:
        place = ''.join(filter(str.isalpha, folder.split('-')[0]))
        level = int(folder.split('-')[-1])
    except ValueError:
        print(f"Skipping folder {folder}, invalid format.")
        continue

    # 检查progress.json文件是否存在
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)

        # 检查origin, spe, sub是否相等
        if progress_data["origin"] == progress_data["spe"] == progress_data["sub"]:
            # 读取final_merge_df_all_batches.pkl文件
            pkl_file = os.path.join(data_folder, "final_merge_df_all_batches.pkl")
            if os.path.exists(pkl_file):
                df = pd.read_pickle(pkl_file)
                # 添加place和level列
                df['place'] = place
                df['level'] = level
                all_dfs.append(df)
                print(f"Added data from {pkl_file} in folder {folder}")
            else:
                print(f"{pkl_file} does not exist in folder {folder}.")
        else:
            print(f"origin, spe, and sub are not equal in folder {folder}. No action needed.")
    else:
        print(f"Progress file {progress_file} does not exist in folder {folder}.")
