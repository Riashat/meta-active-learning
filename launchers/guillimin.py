"""
This file creates PBS compatible jobs
sends them to the server.
"""

import json
from itertools import product
import argparse
import sys, os
from subprocess import run

create_folder = lambda f: [ os.makedirs(f) if not os.path.exists(f) else False ]

# locaiton of the project
HERE = os.path.realpath(
                os.path.join(
                    os.path.realpath(__file__),
                    '..', # out of the file ./guillimin.py
                    '..')) # out of the folder ./launchers

RUNNABLE = os.path.join(PROJECT_ROOT, './experiment.py')

HEADERS = """#!/bin/bash
#PBS -l nodes=1:ppn={nodes}{gpu}
#PBS -l walltime={walltime}
#PBS -A {group}
#PBS -N {name}
#PBS -o {savefolder}/out.txt
#PBS -e {savefolder}/error.txt
#PBS -q {queue}
"""

MODULES = """cd $PBS_O_WORKDIR
module load foss/2015b
module load Python/3.5.2
module load CUDA_Toolkit/7.5
module load cuDNN/5.0-ga
module load Tensorflow/1.0.0-Python-3.5.2
env

"""

MAIN_COMMAND = "python3.5 {RUNNABLE} --data {dataset} -e {epochs} -p {policy} -a {acquisitions} -d {dropoutiterations} -f {outfolder} --model {} -s {SEED}"

def process_wall_time(args):
    if args.queue == 'debug':
        return '00:30:00'
    elif args.queue == 'k20':
        return '12:00:00' # 12 hours on gpu
    elif args.queue == 'sw':
        return '24:00:00' # 24 hours on cpu
    else:
        raise ValueError('Unknown queue')

def grid_search(args_vals):
    """ arg_vals: a list of lists, each one of format (argument, list of possible values) """
    lists = []
    for arg_vals in args_vals:
        arg, vals = arg_vals
        ll = []
        for val in vals:
            ll.append("-" + arg + " " + str(val) + " ")
        lists.append(ll)
    return ["".join(item) for item in itertools.product(*lists)]

def main(args):
    # ----------------------------------------
    # Prepare bash file template
    # ----------------------------------------

    bashfile = HEADERS + MODULES
    
    command = MAIN_COMMAND
    MAIN_COMMAND += 

    # ----------------------------------------
    # Prepare sequence of parameters
    # ----------------------------------------
    grid = [
            ['p', args.policy],
            ['r', args.reward],
            ['gamma', args.gamma],
            ['policyparam', args.policy_param]
        ] 

    for settings in grid_search(grid):
        # job_str = job_prefix + settings
        # job_strs.append(job_str)
        # for e in range(args.exp):
        #     # ----------------------------------------
        #     # Create folders and files
        #     # ----------------------------------------
        #     parameters = json.dumps(param)
        #     savefolder = args.out+'GS_'+args.architecture+'_'+str(i)
        #     create_folder(savefolder)
        #     savefolder = os.path.realpath(savefolder)
        #     paramsfile = os.path.join(savefolder, 'params.json')
        #     json.dump(param, open(paramsfile, 'w'))
        #     runnable = os.path.join(savefolder+'.sh')
        #     with open(runnable, 'w') as f:
        #         # ----------------------------------------
        #         # Create parameters file
        #         # ----------------------------------------
        #         gpu = ':gpus=1:exclusive_process' if args.queue in ['aw', 'k20'] else ''
        #         f.write(bashfile.format(nodes=args.n_cpus,
        #                                 name=args.architecture+'_'+str(i),
        #                                 gpu=gpu,
        #                                 queue=args.queue,
        #                                 parameters=paramsfile,
        #                                 datafolder=args.folder,
        #                                 savefolder=savefolder,
        #                                 random_seed=args.random_seed,
        #                                 architecture=args.architecture))
        #     pdb.set_trace()
        #     run(['qsub', runnable])
        # pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    job_arguments = parser.add_argument_group('Job Arguments')
    job_arguments.add_argument("-group", "--group", 
                        help="Group allocation ID",
                        required=False, default='ams-754-aa')

    job_arguments.add_argument("-queue", "--queue", 
                        help="""The queue to run use `debug` 
                                for debugging. use `k20` for gpus.
                                Use `sw` for regular cpu jobs""",
                        required=True, type=str)

    job_arguments.add_argument("-ncpus", "--ncpus",
                        help="Number of CPUs to use (16 to use a full GPU)",
                        required=False, default=2, type=int)

    job_arguments.add_argument("-expname", "--expname",
                        help="""The name of the experiment""",
                        required=True, type=str)
    
    job_arguments.add_argument("-dry", "--dry",
                        help="""Dry run""",
                        required=False, type=str, default=True)

    experiment_arguments = parser.add_argument_group('Experiment Arguments')

    experiment_arguments.add_argument('-exp', type=int, default=5, help='number of repeats')
    experiment_arguments.add_argument('-e', type=int, default=1000, help=['number of epochs for training'], nargs='+')
    experiment_arguments.add_argument('-a', '--acquisitions', type=int, default=980, help=['total number of acquisitions'])
    experiment_arguments.add_argument('-d', '--dropoutiterations', type=int, default=100, help='Dropout samples')
    experiment_arguments.add_argument('-m', '--model', type=str, default='bayesian', help=['bayesian', 'deterministic'])
    experiment_arguments.add_argument('-data', '--dataset', type=str, default='mnist')

    gridsearch_arguments = parser.add_argument_group('Arguments to grid search over')

    gridsearch_arguments.add_argument('-p', '--policy', type=str, required=True, help=['bandit-ucb', 'bandit-epsilongreedy', 'random'], nargs='+')
    gridsearch_arguments.add_argument('-r', '--reward', type=str, default='acc', nargs='+')
    gridsearch_arguments.add_argument('-policyparam', '--policy-param', type=float, default=0.5, nargs='+')
    gridsearch_arguments.add_argument('-gamma', '--gamma', required=False, type=float, default=None, nargs='+')

    args = parser.parse_args()
    if args.queue in ['aw', 'k20'] and args.n_cpus < 16:
        print('WARNING: Using less than a complete GPU')
    main(args)
