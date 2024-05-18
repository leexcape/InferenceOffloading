#!/bin/bash

#SBATCH -J VGG_Training
#SBATCH --account=radardqn
#SBATCH --partition=p100_dev_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

module load Anaconda3/2020.11

source activate
source deactivate

conda run -n DL_basic python VGG16_Train.py
