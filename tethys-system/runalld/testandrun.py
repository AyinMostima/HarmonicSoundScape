import os
import json
import subprocess
# 设定目录
base_dir = "/lustre/home/acct-yinshan/yinshan"
target_folder = "voiceprocess2023"
# 找到目标文件夹
target_path = os.path.join(base_dir, target_folder)
# 列出目标文件夹下的所有文件夹
subfolders = [f.name for f in os.scandir(target_path) if f.is_dir()]
print("Found subfolders:", subfolders)
# 遍历所有子文件夹
for folder in subfolders:
    selected_path = os.path.join(target_path, folder)

    # 查看data文件夹中的progress.json文件
    data_folder = os.path.join(selected_path, "data")
    progress_file = os.path.join(data_folder, "progress.json")

    # 检查progress.json文件是否存在
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)

    else:
    # 创建一个默认的进度
        progress_data = {"origin": 0, "spe": 0, "sub": 0}

        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)
        print(f"Created default progress file {progress_file} in folder {folder}.")



    # 检查origin, spe, sub是否相等
    if (not (progress_data["origin"] == progress_data["spe"] == progress_data["sub"])) or ((progress_data["origin"] == 0 and progress_data["spe"] == 0 and progress_data["sub"] == 0)):
            # 执行runanalysis.sh脚本
            run_script = os.path.join(selected_path, "runanalysis.sh")
            run_script_dir = os.path.dirname(run_script)
            if os.path.exists(run_script):
                subprocess.run(["sed", "-i", "s/\r$//", run_script], check=True)
                print(f"Converted line endings for {run_script}")
                subprocess.run(["sbatch", run_script], cwd=run_script_dir, check=True)
                print(f"Executed {run_script} in folder {folder}")
            else:
                print(f"Script {run_script} does not exist in folder {folder}.")
    else:
        print(f"origin, spe, and sub are equal in folder {folder} and not all zero. No action needed.")
