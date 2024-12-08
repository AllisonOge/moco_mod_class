#!/bin/bash
#==============================================
#SBATCH --partition=gpu-v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name=amc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ogechukwukanu@gmail.com
#==============================================

source ~/research/moco_mod_class/venv/bin/activate

python ~/research/moco_mod_class/train.py -c ~/research/moco_mod_class/config.yaml
