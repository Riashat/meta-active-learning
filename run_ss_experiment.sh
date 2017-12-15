#!/bin/bash
#!SBATCH --time=24:00:00
#SBATCH --account=def-steele
#!SBATCH --job-name=sslVAE
#SBATCH --output=sslVAE_out
#SBATCH --error=sslVAE
#!SBATCH --gres=gpu:1  
#!SBATCH --mem=32000M
#SBATCH --mail-user=bogdan.mazoure@mail.mcgill.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


module load cuda cudnn python/3.5.2
module load python/3.5.2
#!module load  python35-mpi4py/2.0.0

cd home/bmazoure
source activate env
cd /project/6006774/bmazoure/algs/meta-active-learning
python3 experiment_ss.py $@

#! sbatch --time=24:00:00 --nodes=1 --gres=gpu:2 --mem=32000M --job-name=sslVAE run_ss_experiment.sh -p uniform-bald -f experiment_ss -data mnist
#! visit https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm for information

