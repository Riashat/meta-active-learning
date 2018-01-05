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

#! sbatch --time=48:00:00 --nodes=1 --gres=gpu:2 --mem=20000M --job-name=sslVAE_gpus_uniform_bald run_ss_experiment_gpu.sh -p uniform-bald -f experiment_ss -data mnist -ts 100 -ps 10000 -e 100 -b 50 -lr 3e-3 -samples 8 --sanity_check 0 --cnn 1 -a 100
#! sbatch --time=48:00:00 --nodes=1 --gres=gpu:2 --mem=20000M --job-name=sslVAE_gpus_random run_ss_experiment_gpu.sh -p random -f experiment_ss -data mnist -ts 100 -ps 10000 -e 100 -b 50 -lr 3e-3 -samples 8 --sanity_check 0 --cnn 1 -a 100
#! sbatch --time=48:00:00 --nodes=1 --gres=gpu:2 --mem=20000M --job-name=sslVAE_gpus_uniform_varratio run_ss_experiment_gpu.sh -p uniform-varratio -f experiment_ss -data mnist -ts 100 -ps 10000 -e 100 -b 50 -lr 3e-3 -samples 8 --sanity_check 0 --cnn 1 -a 100

