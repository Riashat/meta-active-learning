import os
import itertools
import numpy as np
import subprocess
import argparse
parser = argparse.ArgumentParser()

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


parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=int, default=5)
parser.add_argument('-g', required=False, type=str, default='0', help=['specify GPU'])
parser.add_argument('-e', type=int, default=1000, help=['number of epochs for training'])
parser.add_argument('-p', type=str, default='bandit-ucb', help=['bandit-ucb', 'bandit-epsilongreedy', 'random'])
parser.add_argument('-a', type=int, default=980, help=['total number of acquisitions'])
parser.add_argument('-d', type=int, default=100, help='Dropout')
parser.add_argument('-f', type=str, default='.././results/')
parser.add_argument('-m', type=str, default='bayesian', help=['bayesian', 'deterministic'])
parser.add_argument('-r', type=str, default='acc')
parser.add_argument('-policyparam', type=int, default=0.5)
parser.add_argument('-gamma', required=False, type=float, default=None)

locals().update(parser.parse_args().__dict__)    


job_prefix = "python "
exp_script = '../experiment.py ' 
job_prefix += exp_script

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.g

acquisitions = args.a
epochs = args.e
save_dir = args.f
mc_samples = args.d
reward = args.r
model = args.m
policy = args.p
gamma = args.gamma
policyparameters = args.policyparam

experiments = args.exp
save_result = save_dir + 'Rwd-' + reward


grid = [] 
grid += [['e', [epochs]]] 
grid += [['p', [policy]]] # TO DO, to run for all policies at once : ['bandit-ucb', 'bandit-epsilongreedy']
grid += [['a', [acquisitions]]]
grid += [['d', [mc_samples]]]
grid += [['f', [save_result]]]
grid += [['m', [model]]]
grid += [['r', [reward]]]
grid += [['gamma', [gamma]]]
grid += [['policyparam', [policyparameters]]]



job_strs = []
for settings in grid_search(grid):
	for e in range(experiments):	
		job_str = job_prefix + settings
		job_strs.append(job_str)
print("njobs", len(job_strs))

for job_str in job_strs:
    os.system(job_str)

