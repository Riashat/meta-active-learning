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