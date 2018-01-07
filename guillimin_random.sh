#!/bin/bash
#PBS -l nodes=1:ppn=16:gpus=1
#PBS -l walltime=72:00:00
#PBS -A xzv-031-ab
#PBS -N bayesianAL-random
#PBS -o ./bayesianAL_out_random.txt
#PBS -e ./bayesianAL_error_random.txt
#PBS -q aw

cd $PBS_O_WORKDIR

module load foss/2015b
module load Python/3.5.2
module load CUDA_Toolkit/7.5
module load cuDNN/5.0-ga
module load Tensorflow/1.0.0-Python-3.5.2

python3 experiment_ss.py -p random -f experiment_ss -data mnist -ts 100 -ps 10000 -e 100 -b 50 -lr 3e-3 -samples 8 --sanity_check 0 --cnn 1 -a 100