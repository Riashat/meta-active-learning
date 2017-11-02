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

RUNNABLE = os.path.join(HERE, './experiment.py')

HEADERS = """#!/bin/bash
#PBS -l nodes=1:ppn={nodes}{gpu}
#PBS -l walltime={walltime}
#PBS -A {group}
#PBS -N {name}
#PBS -o {outfolder}/out-{settings}.txt
#PBS -e {outfolder}/error-{settings}.txt
#PBS -q {queue}
"""

MODULES = """cd $PBS_O_WORKDIR
module load foss/2015b
module load Python/3.5.2
module load CUDA_Toolkit/7.5
module load cuDNN/5.0-ga
module load Tensorflow/1.0.0-Python-3.5.2
# install an old version of keras
pip3 install old/keras --user
env

"""
# -g gm is basically run on guillimin!
# MAIN_COMMAND = "python3.5 {RUNNABLE} -g gm --data {dataset} -e {epochs} -p {policy} "
# MAIN_COMMAND += "-a {acquisitions} -d {dropoutiterations} -f {outfolder} --model {model} -s {SEED}"


# MAIN_COMMAND = "python3.5 {RUNNABLE} -g gm --data {dataset} -e {epochs} -p {policy} "
# MAIN_COMMAND += "-a {acquisitions} -d {dropoutiterations} -f {outfolder} --model {model} -s {SEED}"

MAIN_COMMAND = """
python3.5 {RUNNABLE} -g gm --data {dataset} -e {epochs} -a {acquisitions} -d {dropoutiterations} -f {outfolder} --model {model}"""

def process_wall_time(args):
    if args.queue == 'debug':
        return '00:30:00'
    elif args.queue == 'k20' or args.queue == 'aw':
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
        print(vals)
        for val in vals:
            ll.append("-" + arg + " " + str(val) + " ")
        lists.append(ll)
    return ["".join(item) for item in product(*lists)]

def main(args):
    # ----------------------------------------
    # Prepare sequence of parameters
    # ----------------------------------------
    grid = [
            ['p', args.policy],
            ['r', args.reward],
            ['w', args.weight_decay],
            ['gamma', args.gamma],
            ['policyparam', args.policy_param]
        ] 


    RESULTS_FOLDER = os.path.join(HERE, args.expname)
    create_folder(RESULTS_FOLDER)
    gpu = ':gpus=1:exclusive_process' if args.queue in ['aw', 'k20'] else ''
    for settings in grid_search(grid):
        bashfile = HEADERS + MODULES
        for e in range(args.exp):
            # create the bash file for the number of replicates we have:
            # setting different seeds at every call.
            bashfile += MAIN_COMMAND + ' '+settings+' -s '+str(e)

        runnable = os.path.join(RESULTS_FOLDER, settings.replace(' ', '')+'.sh')
        
        formatted = bashfile.format(RUNNABLE=RUNNABLE,
                                    acquisitions=args.acquisitions,
                                    dropoutiterations=args.dropoutiterations,
                                    dataset=args.dataset,
                                    outfolder=RESULTS_FOLDER,
                                    epochs=args.epochs,
                                    model=args.model,
                                    nodes=args.ncpus,
                                    gpu=gpu,
                                    walltime=process_wall_time(args),
                                    settings=settings.replace(' ', ''),
                                    group=args.group,
                                    name=args.expname+'-'+settings.replace(' ', ''),
                                    queue=args.queue
                                    )
        if not args.dry:
            with open(runnable, 'w') as f:
                # ----------------------------------------
                # Create parameters file
                # ----------------------------------------

                f.write(formatted)
                run(['qsub', runnable])
        else:
            print('NEXT FILE')
            print(formatted)
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    job_arguments = parser.add_argument_group('Job Arguments')
    job_arguments.add_argument("-group", "--group", 
                        help="Group allocation ID",
                        required=False, default='xzv-031-ab')

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
                        required=False, action='store_true', default=False)

    experiment_arguments = parser.add_argument_group('Experiment Arguments')

    experiment_arguments.add_argument('-exp', type=int, default=5, help='number of repeats')
    experiment_arguments.add_argument('-e', '--epochs', type=int, default=50, help=['number of epochs for training'])
    experiment_arguments.add_argument('-a', '--acquisitions', type=int, default=98, help=['total number of acquisitions'])
    experiment_arguments.add_argument('-d', '--dropoutiterations', type=int, default=50, help='Dropout samples')
    experiment_arguments.add_argument('-m', '--model', type=str, default='bayesian', help=['bayesian', 'deterministic'])
    experiment_arguments.add_argument('-data', '--dataset', type=str, default='mnist')
    experiment_arguments.add_argument('-o', '--other', type=str, default='',
                    help="""other arguments enclosed by double quotes to pass to the script.
                    This must be something like `-o "EXTRA ARGUMENTS"`""")

    gridsearch_arguments = parser.add_argument_group('Arguments to grid search over')

    gridsearch_arguments.add_argument('-p', '--policy', type=str, required=True, help=['bandit-ucb', 'bandit-epsilongreedy', 'random'], nargs='+')
    gridsearch_arguments.add_argument('-r', '--reward', type=str, default=['marginalacc'], nargs='+')
    gridsearch_arguments.add_argument('-policyparam', '--policy-param', type=float, default=[0.5], nargs='+')
    gridsearch_arguments.add_argument('-gamma', '--gamma', required=False, type=float, default=[0.85], nargs='+')
    gridsearch_arguments.add_argument('-w', '--weight_decay', required=False, type=float, default=[3], nargs='+')

    args = parser.parse_args()
    if args.queue in ['aw', 'k20'] and args.n_cpus < 16:
        print('WARNING: Using less than a complete GPU')
    main(args)
