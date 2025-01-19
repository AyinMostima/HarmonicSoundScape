import os
import pandas as pd
import time
import datetime
import json
# Merge all batches from batch directories


# 读取配置文件
with open('config.json', 'r') as file:
    config = json.load(file)
# 从配置文件获取参数
datapath = config['datapath']

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