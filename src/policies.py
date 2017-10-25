import random

class Policy(object):
	def __init__(self, acquisition_functions):
		"""
		Abstract policy for selecting acquisition functions
		"""
		self.acquisition_functions = acquisition_functions

	def get_acquisition_function(self, *args, **kwargs):
		NotImplementedError('Policy Abstract Class has no getter')

	def update_policy(self, *args, **kwargs):
		# @TODO: Ask Riashat how to unify this function call
		NotImplementedError('Policy Abstract Class has no updater')


class UniformPolicy(Policy):
	def __init__(self, acquisition_functions):
		assert len(acquisition_functions) == 1, 'Uniform policy can only have one acquisition function'
		super().__init__(acquisition_functions)

	def get_acquisition_function(self, *args, **kwargs):
		return self.acquisition_functions[0]

	def update_policy(self, *args, **kwargs):
		pass


class RandomPolicy(Policy):
	"""
		Returns a random acquisition function when called
	"""
	def get_acquisition_function(self, *args, **kwargs):
		return random.choice(self.acquisition_functions)

	def update_policy(self, *args, **kwargs):
		return None


class BanditPolicy(Policy):

	def get_acquisition_function(self, *args, **kwargs):
		# return acquisition based on some bandit selection
		pass

	def update_policy(self, *args, **kwargs):
		# update the counts or samplings etc.
		pass


def policy_parser(policy_name):
	from src.acquisition_function import ACQUISITION_FUNCTIONS_TEXT

	if policy_name == 'random':
		print('Random policy where at every step we pick randomly from ')
		print(ACQUISITION_FUNCTIONS_TEXT)
		return RandomPolicy(ACQUISITION_FUNCTIONS_TEXT)
	
	elif policy_name.startswith('uniform-'):
		print('Uniform policy where at every step the acquisition function is')
		acquisition_function = policy_name.replace('uniform-', '')
		print(acquisition_function)
		assert acquisition_function in ACQUISITION_FUNCTIONS_TEXT
		return UniformPolicy([acquisition_function])
	
	elif policy_name == 'bandit':
		print('Learned policy where at every step we pick according to a policy over ')
		print(ACQUISITION_FUNCTIONS_TEXT)

		raise NotImplementedError('Bandit policy not implemented yet')

