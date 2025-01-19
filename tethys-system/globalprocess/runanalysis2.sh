#!/bin/bash
#SBATCH --job-name=runcal      
#SBATCH --partition=192c6t     
#SBATCH -n 60                 
#SBATCH --ntasks-per-node=60   
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3
source activate /lustre/home/acct-yinshan/yinshan/.conda/envs/voiceprocgpu
python globaldatasetprocess.py