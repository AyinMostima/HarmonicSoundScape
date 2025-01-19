#!/bin/bash
#SBATCH --job-name=merge2       
#SBATCH --partition=huge      
#SBATCH -n 8 
#SBATCH --ntasks-per-node=8
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3
source activate /lustre/home/acct-yinshan/yinshan/.conda/envs/voiceprocgpu/envs/dataanalysis
python mergeasasheetbatch.py
