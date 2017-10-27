from src.policies import policy_parser
from collections import namedtuple
import numpy as np

argument_holder = namedtuple('args', ['policy_param', 'gamma'])
policy_arguments = argument_holder(policy_param=0.5, gamma=0.5)


def run_policy(policy):

	for i in range(4000):
		acquisition_function = policy.get_acquisition_function()
		if acquisition_function == 'bald':
			reward = 5
		else:
			reward = np.random.random()

		policy.update_policy(reward)

def count_balds(policy):
	bald_count = 0
	for i in range(50):
		if policy.get_acquisition_function() == 'bald':
			bald_count += 1

	return bald_count / 50.0

def test_ucb_policy():
	policy = policy_parser('bandit-ucb', policy_arguments)
	assert count_balds(policy) < 0.5

	policy = policy_parser('bandit-ucb', policy_arguments)
	run_policy(policy)
	assert count_balds(policy) > 0.9 


def test_eps_policy():
	policy = policy_parser('bandit-epsilongreedy', policy_arguments)
	run_policy(policy)
	assert count_balds(policy) > 0.25 

def test_uniform_policy():
	policy = policy_parser('uniform-bald', policy_arguments)
	assert policy.get_acquisition_function() == 'bald'
	run_policy(policy)
	assert policy.get_acquisition_function() == 'bald'