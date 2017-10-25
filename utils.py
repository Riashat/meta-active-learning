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
            required=False, type=int, default=1000)

      named_args.add_argument('--policy',
            help="""Policy for selecting acquisition functions : 'random', 'uniform-*'""",
            required=True, type=str, default='random')

      named_args.add_argument('-a', '--acquisitions',
            help="""# of acquisitions for active learning""",
            required=False, type=int, default=980)

      named_args.add_argument('-d', '--dropoutiterations',
            help="""# of dropout estimates""",
            required=False, type=int, default=100)

      named_args.add_argument('-f', '--folder',
            help="""Folder to save data to""",
            required=True, type=str, default='./results/')
      
      named_args.add_argument('-s', '--seed',
            help="""Random seed to use""",
            required=False, type=int, default=25102017)
      return parser


class Logger(object):
      def __init__(self, experiment_name='', folder='./results'):
            self.train_loss = []
            self.val_loss = []
            self.train_acc = []
            self.val_acc = []
            self.save_folder = os.path.join(folder, experiment_name+time.strftime('%y-%m-%d-%H-%M-%s'))
            create_folder(save_folder)

      def record_train_metrics(self, train_loss, train_acc):
            self.train_acc.append(train_acc)
            self.train_loss.append(train_loss)

      def record_val_metrics(self, val_loss, val_acc):
            self.val_acc.append(val_acc)
            self.val_loss.append(val_loss)

      def save(self):
            np.save(os.path.join(self.save_folder, "train_loss.npy"), self.train_loss)
            np.save(os.path.join(self.save_folder, "val_loss.npy"), self.val_loss)
            np.save(os.path.join(self.save_folder, "train_acc.npy"), self.train_acc)
            np.save(os.path.join(self.save_folder, "val_acc.npy"), self.val_acc)

      def save_args(self, args):
            with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
                  json.dump(dict(args._get_kwargs(), f)
