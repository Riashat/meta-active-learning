import random
import numpy as np

class Policy(object):
    def __init__(self, acquisition_functions):
        """
        Abstract policy for selecting acquisition functions
        """
        self.acquisition_functions = acquisition_functions

    def get_acquisition_function(self, *args, **kwargs):
        raise NotImplementedError('Policy Abstract Class has no getter')

    def update_policy(self, reward, action=None):
        print('No policy to update!')
        pass

class UniformPolicy(Policy):
    def __init__(self, acquisition_functions):
        assert len(acquisition_functions) == 1, 'Uniform policy can only have one acquisition function'
        super().__init__(acquisition_functions)

    def get_acquisition_function(self, *args, **kwargs):
        return self.acquisition_functions[0]

class RandomPolicy(Policy):
    """
    Returns a random acquisition function when called
    """
    def get_acquisition_function(self, *args, **kwargs):
        return random.choice(self.acquisition_functions)

class BanditPolicy(Policy):
    """
    Model the rewards from acquisition functions
    as a Bandit.
    Note that this is just the skeleton and doesn't
    actually do anything, you must use EpsilonGreedyBanditPolicy
    or UCBBanditPolicy to define the explore-exploit strategy
    """
    def __init__(self,
                 acquisition_functions,
                 gamma=None,
                 prior=0):

        super().__init__(acquisition_functions)
        self.k = len(self.acquisition_functions)
        self.prior = prior
        self.gamma = gamma
        self._value_estimates = prior*np.ones(self.k)
        self.action_attempts = np.zeros(self.k)
        self.t = 0
        self.last_action = None

    def internal_policy(self):
        """
        Implement 
        """
        raise NotImplementedError('No policy defined for arm picking')

    def get_acquisition_function(self):
        action = self.internal_policy()
        self.last_action = action
        return self.acquisition_functions[action]

    def update_policy(self, reward, verbose=False):
        self.action_attempts[self.last_action] += 1

        if self.gamma is None:
            g = 1.0 / self.action_attempts[self.last_action]
        else:
            g = self.gamma
        q = self._value_estimates[self.last_action]

        self._value_estimates[self.last_action] += g*(reward - q)
        self.t += 1

        if verbose:
            print('Policy updated')

class EpsilonGreedyBanditPolicy(BanditPolicy):
    """
    Choose the acquisition functions according to an 
    epsilon greedy policy
    """
    def __init__(self,
                 acquisition_functions,
                 gamma=None,
                 prior=0,
                 epsilon=0.5):
        super().__init__(acquisition_functions, gamma=gamma, prior=prior)
        self.epsilon = epsilon

    def internal_policy(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self._value_estimates))
        else:
            action = np.argmax(self._value_estimates)
            check = np.where(self._value_estimates == action)[0]
            if len(check) == 0:
                return action
            else:
                return np.random.choice(check)


class UCBBanditPolicy(BanditPolicy):
    def __init__(self,
                 acquisition_functions,
                 gamma=None,
                 prior=0,
                 c=0.5):
        super().__init__(acquisition_functions, gamma=gamma, prior=prior)
        self.c = c

    def internal_policy(self):
        exploration = np.log(self.t+1) / self.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.c)

        q = self._value_estimates + exploration
        action = np.argmax(q)
        check = np.where(q == action)[0]
        if len(check) == 0:
            return action
        else:
            return np.random.choice(check)


def policy_parser(policy_name, args):
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

    elif policy_name.startswith('bandit-'):
        print('At every step we pick actions from ')
        print(ACQUISITION_FUNCTIONS_TEXT)
        policy = policy_name.replace('bandit-', '')
        print('according to a policy')
        print(policy)
        if policy == 'ucb':
            return UCBBanditPolicy(ACQUISITION_FUNCTIONS_TEXT,
                                   gamma=args.gamma,
                                   c=args.policy_param)
        elif policy == 'epsilongreedy':
            return EpsilonGreedyBanditPolicy(ACQUISITION_FUNCTIONS_TEXT,
                                             gamma=args.gamma,
                                             epsilon=args.policy_param)
