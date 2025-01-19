#!/bin/bash
#SBATCH --job-name=merge2       
#SBATCH --partition=cpu        
#SBATCH -n 40               
#SBATCH --ntasks-per-node=40 
#SBATCH --output=%j.out
#SBATCH --error=%j.err


module load miniconda3
source activate /lustre/home/acct-yinshan/yinshan/.conda/envs/voiceprocgpu/envs/dataanalysis
python mergeasasheetbatch.py
