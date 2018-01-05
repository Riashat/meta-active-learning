#!/bin/bash

sbatch --time=72:00:00 --nodes=1 --gres=gpu:1 --mem=10000M --job-name=sslVAE_gpus_uniform_bald run_ss_experiment_gpu.sh -p uniform-bald -f experiment_ss -data mnist -ts 100 -ps 10000 -e 100 -b 50 -lr 3e-3 -samples 8 --sanity_check 0 --cnn 1 -a 100
sbatch --time=72:00:00 --nodes=1 --gres=gpu:1 --mem=10000M --job-name=sslVAE_gpus_random run_ss_experiment_gpu.sh -p random -f experiment_ss -data mnist -ts 100 -ps 10000 -e 100 -b 50 -lr 3e-3 -samples 8 --sanity_check 0 --cnn 1 -a 100
sbatch --time=72:00:00 --nodes=1 --gres=gpu:1 --mem=10000M --job-name=sslVAE_gpus_uniform_varratio run_ss_experiment_gpu.sh -p uniform-varratio -f experiment_ss -data mnist -ts 100 -ps 10000 -e 100 -b 50 -lr 3e-3 -samples 8 --sanity_check 0 --cnn 1 -a 100