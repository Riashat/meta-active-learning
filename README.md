# Meta-Learning Bayesian Active Learning Algorithms
Using Bandits to learn a policy to select acquisition functions

## Requirements
Python 3.5.2

See `requirements.txt`

You can install as `pip3.5 install -r requirements.txt --user`

## Tests
Some unit tests exist
run `python3 -m pytest` on linux or `pytest` on os x.

## Running

```
usage: experiment.py [-h] [-g GPU] [-e EPOCHS] -p POLICY [-a ACQUISITIONS]
                     [-d DROPOUTITERATIONS] -f FOLDER [-s SEED] [-m MODEL]
                     [-r REWARD] [-gamma GAMMA] [-policyparam POLICY_PARAM]

optional arguments:
  -h, --help            show this help message and exit

named arguments:
  -g GPU, --gpu GPU     gpu to use
  -e EPOCHS, --epochs EPOCHS
                        # of epochs to train
  -p POLICY, --policy POLICY
                        Policy for selecting acquisition functions : 'random',
                        'uniform-*' or 'bandit-*' uniform-* where * can be:
                        any of the acquisition functions implemented
                        (maxentropy, segnet, bald etc..) bandit-* where * is
                        the policy to play can be ucb for upper confidence
                        bound or epsilongreedy for epsilon greedy algorithm
  -a ACQUISITIONS, --acquisitions ACQUISITIONS
                        # of acquisitions for active learning
  -d DROPOUTITERATIONS, --dropoutiterations DROPOUTITERATIONS
                        # of dropout estimates
  -f FOLDER, --folder FOLDER
                        Folder to save data to
  -s SEED, --seed SEED  Random seed to use
  -m MODEL, --model MODEL
                        Model to use: `bayesian` or `regular`
  -r REWARD, --reward REWARD
                        Reward to use: `marginalacc`, `marginallogp`, `logp`,
                        `acc`
  -gamma GAMMA, --gamma GAMMA
                        The gamma discount factor to use
  -policyparam POLICY_PARAM, --policy-param POLICY_PARAM
                        This is either epislon or c depending on which bandit
                        policy you chose

```

## Available Policies

1. `random`: randomly pick a different acqusition function at each round
2. `uniform-maxentropy`: use `maxentropy` acquisition function for the whole experiment
1. `uniform-segnet`: use `segnet` acquisition function for the whole experiment
1. `uniform-bald`: use `bald` acquisition function for the whole experiment
1. `uniform-varratio`: use `varratio` acquisition function for the whole experiment
1. `bandit-ucb`: Use UCB to learn which acquisition functions to pick
2. `bandit-epsilongreedy`: use epsilon greedy to learn which acquisition functions to pick


For example:

```
python3.5 experiment.py -e 100 -p bandit-epsilongreedy -policyparam 0.2 -gamma 0.9 -r 'acc' -a 10 -d 10 -m bayesian
```

will run a bayesian CNN for 100 epochs using an epsilongreedy policy with epsilon 0.2 and discount factor 0.9 with reward as the validation accuracy. We will querey the policy 10 times and use 10 dropout iterations to estimate the uncertainty of pool points.


# Creating plots.

```
usage: create_plots.py [-h] -f FOLDERS [FOLDERS ...] -m METRICS [METRICS ...]
                       -name NAME

optional arguments:
  -h, --help            show this help message and exit

named arguments:
  -f FOLDERS [FOLDERS ...], --folders FOLDERS [FOLDERS ...]
                        folders with the experiments. This must be can be
                        follows: -f FOLDER1 FOLDER2 where FOLDER1 and FOLDER2
                        are two experiments each with subfolders with
                        replicates: FOLDER1/ replicate1 replicate2 FOLDER2/
                        replicate1 replicate2
  -m METRICS [METRICS ...], --metrics METRICS [METRICS ...]
                        metrics to plot. can be multiple metrics like: -m
                        val_acc train_acc or must be `-m acq` for plotting
                        acquition functions
  -name NAME, --name NAME
                        title of figure, will be saved as NAME.pdf

```


```
python3.5 create_plots.py -f './results/experiment1' 'results/experiment2' -m 'val_acc' 'train_acc' -n 'accuracy_plots'
```

This will create a figure called `accuracy_plots.pdf` which will plot the `val_acc`s and `train_acc`s from `experiment1` and `experiment2`


It is also possible to plot the acquisition function used in each scheme as follows:

```
python3.5 create_plots.py -f './results/experiment1' 'results/experiment2' -m acq -n 'acq curves'
```
