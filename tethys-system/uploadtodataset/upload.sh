#!/bin/bash
#SBATCH --job-name=upload       
#SBATCH --partition=cpu        
#SBATCH -n 40               
#SBATCH --ntasks-per-node=40 
#SBATCH --output=%j.out
#SBATCH --error=%j.err


# 设置环境变量以使用代理
export http_proxy=http://proxy2.pi.sjtu.edu.cn:3128
export https_proxy=http://proxy2.pi.sjtu.edu.cn:3128
export no_proxy="localhost,127.0.0.1,localaddress,.localdomain.com"

module load miniconda3
source activate /lustre/home/acct-yinshan/yinshan/.conda/envs/voiceprocgpu/envs/dataanalysis
python uploadtosql.py
