#!/bin/bash
#!SBATCH --time=24:00:00
#SBATCH --account=def-steele
#!SBATCH --job-name=sslVAE_gpus
#SBATCH --output=sslVAE_out_gpus
#SBATCH --error=sslVAE_err_gpus
#!SBATCH --gres=gpu:2  
#!SBATCH --mem=32000M
#SBATCH --mail-user=bogdan.mazoure@mail.mcgill.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


module load cuda cudnn python/3.5.2
module load python/3.5.2
#!module load  python35-mpi4py/2.0.0

cd /home/bmazoure
source activate env
cd /project/6006774/bmazoure/algs/meta-active-learning
python3 experiment_ss.py $@

#! sbatch --time=24:00:00 --kill-on-invalid-dep=yes --nodes=2 --gres=gpu:2 --mem=32000M --job-name=sslVAE_gpus run_ss_experiment_gpu.sh -p uniform-bald -f experiment_ss -data mnist -ts 1000 -ps 10000 -e 200 -b 50 -lr 3e-3 -samples 8

