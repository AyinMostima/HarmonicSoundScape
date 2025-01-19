import os
import json
import subprocess

# 设定目录
base_dir = "/lustre/home/acct-yinshan/yinshan"
target_folder = "voiceprocess2024"
output_folder = os.path.join(base_dir, "voicesheetoutput")
checkpoint_file = os.path.join(output_folder, "singleremergedone.json")

# 创建输出目录（如果不存在）
os.makedirs(output_folder, exist_ok=True)


# 找到目标文件夹
target_path = os.path.join(base_dir, target_folder)
# 列出目标文件夹下的所有文件夹
subfolders = [f.name for f in os.scandir(target_path) if f.is_dir()]
print("Found subfolders:", subfolders)

# 读取检查点文件
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        processed_folders = json.load(f)
else:
    processed_folders = []

# 遍历所有子文件夹
for folder in subfolders:
    if folder in processed_folders:
        print(f"Skipping folder {folder}, already processed.")
        continue

    selected_path = os.path.join(target_path, folder)

    # 查看data文件夹中的progress.json文件
    data_folder = os.path.join(selected_path, "data")
    progress_file = os.path.join(data_folder, "progress.json")

    # 检查progress.json文件是否存在
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)

        # 检查origin, spe, sub是否相等
        if progress_data["origin"] == progress_data["spe"] == progress_data["sub"]:
            # 执行mergeall.py脚本
            merge_script = os.path.join(selected_path, "mergeall.py")
            if os.path.exists(merge_script):
                try:
                    subprocess.run(["python", merge_script], cwd=selected_path, check=True)
                    print(f"Executed {merge_script} in folder {folder} successfully.")

                    # 更新检查点文件
                    processed_folders.append(folder)
                    with open(checkpoint_file, 'w') as f:
                        json.dump(processed_folders, f)
                except subprocess.CalledProcessError as e:
                    print(f"Error executing {merge_script} in folder {folder}: {e}")
            else:
                print(f"Script {merge_script} does not exist in folder {folder}.")
        else:
            print(f"origin, spe, and sub are not equal in folder {folder}. No action needed.")
    else:
        print(f"Progress file {progress_file} does not exist in folder {folder}.")
