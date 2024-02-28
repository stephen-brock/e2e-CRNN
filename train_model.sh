#!/usr/bin/env bash
#SBATCH --job-name=cw
#SBATCH --account=COMS030144
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem 4GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

#REMOVE --spectrogram IF UNABLE TO COMPILE ON BLUECRYSTAL
python train_model.py --length 256 --stride 256 --epochs 40 --learning-rate 7e-4 --gamma 0.95 --dropout --norm
