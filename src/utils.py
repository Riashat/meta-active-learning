import argparse
import numpy as np
import random
import os
import time
import json

create_folder = lambda f: [ os.makedirs(f) if not os.path.exists(f) else False ]

def get_parser():
      parser = argparse.ArgumentParser()
      named_args = parser.add_argument_group('named arguments')

      named_args.add_argument('-g', '--gpu',
                              help="""gpu to use""",
                              required=False, type=str, default='0')

      named_args.add_argument('-e', '--epochs',
            help="""# of epochs to train""",
            required=False, type=int, default=50)

      named_args.add_argument('-p', '--policy',
            help="""Policy for selecting acquisition functions : 
                    'random', 'uniform-*' or 'bandit-*'
                    
                    uniform-* where * can be:
                        any of the acquisition functions implemented
                        (maxentropy, segnet, bald etc..)

                    bandit-* where * is the policy to play
                        can be ucb for upper confidence bound
                        or epsilongreedy for epsilon greedy algorithm
                    """,
            required=True, type=str, default='random')

      named_args.add_argument('-a', '--acquisitions',
            help="""# of acquisitions for active learning""",
            required=False, type=int, default=98)

      named_args.add_argument('-d', '--dropoutiterations',
            help="""# of dropout estimates""",
            required=False, type=int, default=100)

      named_args.add_argument('-f', '--folder',
            help="""Folder to save data to""",
            required=True, type=str, default='./results/')
      
      named_args.add_argument('-s', '--seed',
            help="""Random seed to use""",
            required=False, type=int, default=25102017)
      
      named_args.add_argument('-m', '--model',
            help="""Model to use: `bayesian` or `regular`""",
            required=False, type=str, default='bayesian')
      
      named_args.add_argument('-r', '--reward',
            help="""Reward to use: `marginalacc`, `marginallogp`, `logp`, `acc`""",
            required=False, type=str, default='marginalacc')

      named_args.add_argument('-data', '--data',
            help="""the data to use, `mnist` and `cifar10` supported. mnist is default""",
            required=False, type=str, default='mnist')

      named_args.add_argument('-gamma', '--gamma',
            help="""The gamma discount factor to use""",
            required=False, type=float, default=None)

      named_args.add_argument('-policyparam', '--policy-param',
            help="""This is either epislon or c depending on which
                    bandit policy you chose""",
            required=False, type=float, default=0.5)  
      named_args.add_argument('-w', '--weight_decay',
            help="""Weight Decay for the L2 regularizer""",
            required=False, type=float, default=3)                  
      
      named_args.add_argument('-b', '--batch_size',
            help="""Batch Size""",
            required=False, type=int, default=128)                  
      
      named_args.add_argument('-q', '--queries',
            help="""Queries from Pool Set""",
            required=False, type=int, default=10)

      named_args.add_argument('-ts', '--training_size',
            help="""Size of the training set""",
            required=False, type=int, default=1000)   

      named_args.add_argument('-ps', '--pool_size',
            help="""Size of the pool set""",
            required=False, type=int, default=5000)

      named_args.add_argument('-lr', '--lr',
            help="""Learning rate""",
            required=False, type=float, default=3e-3)

      named_args.add_argument('-samples', '--samples',
            help="""Number of Monte Carlo samples""",
            required=False, type=int, default=8)

      named_args.add_argument('-check', '--sanity_check',
            help="""Try various training set sizes in sanity check mode""",
            required=False, type=int, default=0)

      named_args.add_argument('-custom', '--custom',
            help="""
               custom acquisition functions to use:
               Here are some defaults:
                  3arm-trivial: [bald, random, negative_bald]
                  all: [] 
               Otherwise you can access any combination 
               of acqusition functions you want by just passing their names
            """,
            required=False, nargs='+', type=str, default='all')      
            
      return parser


class Logger(object):
      def __init__(self, experiment_name='', folder='./results'):
            """
            Saves experimental metrics for use later.
            :param experiment_name: name of the experiment
            :param folder: location to save data
            """
            self.train_loss = []
            self.val_loss = []
            self.train_acc = []
            self.val_acc = []
            self.test_loss = []
            self.test_acc = []
            self.acquisition_functions_used = []
            self.rewards = []
            self.save_folder = os.path.join(folder, experiment_name, time.strftime('%y-%m-%d-%H-%M-%s'))
            create_folder(self.save_folder)

      def record_train_metrics(self, train_loss, train_acc):
            """
            Record training metrics
            """
            self.train_acc.append(train_acc)
            self.train_loss.append(train_loss)

      def record_reward(self, reward):
            self.rewards.append(reward)

      def record_acquisition_function(self, acquisition_function_used):
            self.acquisition_functions_used.append(acquisition_function_used)

      def record_val_metrics(self, val_loss, val_acc):
            """
            Record validation metrics
            """
            self.val_acc.append(val_acc)
            self.val_loss.append(val_loss)

      def record_test_metrics(self, test_loss, test_acc):
            """
            Record test metrics
            """
            self.test_acc.append(test_acc)
            self.test_loss.append(test_loss)

      def save(self):
            np.save(os.path.join(self.save_folder, "train_loss.npy"), self.train_loss)
            np.save(os.path.join(self.save_folder, "val_loss.npy"), self.val_loss)
            np.save(os.path.join(self.save_folder, "train_acc.npy"), self.train_acc)
            np.save(os.path.join(self.save_folder, "val_acc.npy"), self.val_acc)
            np.save(os.path.join(self.save_folder, "test_acc.npy"), self.test_acc)
            np.save(os.path.join(self.save_folder, "test_loss.npy"), self.test_loss)
            np.save(os.path.join(self.save_folder, "rewards.npy"), self.rewards)

            with open(os.path.join(self.save_folder, "acqusition_function_history.txt"), 'w') as f:
                  for acq in self.acquisition_functions_used:
                        f.write(acq+'\n')

      def save_args(self, args):
            """
            Save the command line arguments
            """
            with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
                  json.dump(dict(args._get_kwargs()), f)


class RewardProcess(object):
      def __init__(self, reward_type):
            """
            A unified interface for dealing with the reward process
            """
            print('Reward process initialized using:', reward_type)
            if reward_type == 'marginalacc':
                  def get_reward(previous_accuracy,
                                 current_accuracy,
                                 previous_logp,
                                 current_logp):
                        return current_accuracy - previous_accuracy
            elif reward_type == 'logp':
                  def get_reward(previous_accuracy,
                                 current_accuracy,
                                 previous_logp,
                                 current_logp):
                        return current_logp
            elif reward_type == 'marginallogp':
                  def get_reward(previous_accuracy,
                                 current_accuracy,
                                 previous_logp,
                                 current_logp):
                        return previous_logp - current_logp
            elif reward_type == 'acc':
                  def get_reward(previous_accuracy,
                                 current_accuracy,
                                 previous_logp,
                                 current_logp):
                        return current_accuracy

            self.get_reward = get_reward

def stochastic_evaluate(model, data, n_forward_passes):
      metrics = []
      for i in range(n_forward_passes):
            metrics.append(model.evaluate(*data))
      metrics = np.array(metrics)
      return np.mean(metrics, axis=0).tolist()
