#!/bin/bash
#SBATCH --job-name=merge1       
#SBATCH --partition=192c6t     
#SBATCH -n 60                 
#SBATCH --ntasks-per-node=60   
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3
source activate /lustre/home/acct-yinshan/yinshan/.conda/envs/voiceprocgpu/envs/dataanalysis
python mergeall.py
