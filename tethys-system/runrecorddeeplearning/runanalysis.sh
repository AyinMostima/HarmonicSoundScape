#!/bin/bash
#SBATCH --job-name=runcal      
#SBATCH --partition=cpu        
#SBATCH -n 30                
#SBATCH --ntasks-per-node=30 
#SBATCH --output=%j.out
#SBATCH --error=%j.err


module load miniconda3
source activate /lustre/home/acct-yinshan/yinshan/.conda/envs/voiceprocgpu
python process_calculate.py