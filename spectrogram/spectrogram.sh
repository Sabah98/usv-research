#!/bin/sh
#SBATCH --job-name=spectrogen
#SBATCH --output spectrogen%j.out
#SBATCH --error spectrogen%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p gcai-lab

##Add your code here:

hostname
date
cd /work/sanis/usv-research/spectrogram/
python3 spectrogram.py


