import os
from urllib.parse import urlparse
from sqlalchemy import create_engine, text
import pandas as pd
import json
import numpy as np
import psycopg2

# 设定目录
base_dir = "E:\\"
output_folder = os.path.join(base_dir, "datadownload")
checkpoint_file = os.path.join(output_folder, "sqluploaddone.json")

# 创建数据库连接引擎
db_url =   'postgresql+psycopg2://postgres:123456@localhost:5432/postgres'
engine = create_engine(db_url, isolation_level="AUTOCOMMIT")

# 读取检查点文件
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        processed_files = json.load(f)
else:
    processed_files = []


# 准备数据函数
def prepare_data_for_sql(df):
    """
    Convert all NumPy arrays in the DataFrame to lists to avoid psycopg2.ProgrammingError.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns that may have NumPy array types.

    Returns:
        pd.DataFrame: A new DataFrame with NumPy arrays converted to lists.
    """
    # 创建一个新的 DataFrame，以避免修改原始数据
    new_df = df.copy()

    # 遍历每一列，检查数据类型
    for column in new_df.columns:
        if isinstance(new_df[column].iloc[0], np.ndarray):
            # 如果列中的数据类型是 NumPy 数组，转换为列表
            new_df[column] = new_df[column].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    return new_df


# 遍历所有pkl文件
pkl_files = [f for f in os.listdir(output_folder) if f.endswith('.pkl')]

for pkl_file in pkl_files:
    if pkl_file in processed_files:
        print(f"Skipping file {pkl_file}, already processed.")
        continue

    # 读取pkl文件
    file_path = os.path.join(output_folder, pkl_file)
    df = pd.read_pickle(file_path)

    # 准备数据
    df = prepare_data_for_sql(df)

    # 导入数据
    df.to_sql('audio_data', con=engine, index=False, if_exists='append')
    print(f"Uploaded {pkl_file} to database.")

    # 更新检查点文件
    processed_files.append(pkl_file)
    with open(checkpoint_file, 'w') as f:
        json.dump(processed_files, f)



'''
# 删除重复数据
def remove_duplicates(db_url):
    engine = create_engine(db_url, isolation_level="AUTOCOMMIT")
    delete_duplicates_sql = """
    DELETE FROM audio_data a
    USING audio_data b
    WHERE a.ctid < b.ctid
    AND a.file = b.file
    AND a.level = b.level
    AND a.place = b.place
    """
    with engine.connect() as connection:
        result = connection.execute(text(delete_duplicates_sql))
        print(f"Rows affected: {result.rowcount}")

    print("Duplicate rows deleted from original table.")


# 清理数据库表
def vacuum_table(db_url):
    result = urlparse(db_url)
    username = result.username
    password = result.password
    database = result.path[1:]  # 去掉开头的 '/'
    hostname = result.hostname
    port = result.port
    dsn = f"dbname={database} user={username} password={password} host={hostname} port={port}"
    conn = psycopg2.connect(dsn)
    conn.autocommit = True  # 设置为自动提交模式，确保 VACUUM 不在事务中运行
    try:
        with conn.cursor() as cursor:
            cursor.execute("VACUUM FULL")
            print("Table vacuumed to reclaim space and update statistics.")
    finally:
        conn.close()


# 执行删除重复数据和清理表操作
remove_duplicates(db_url)
vacuum_table(db_url)
'''