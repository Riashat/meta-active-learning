import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from glob import glob

sns.set_style('white')

COLORS = lambda n: list(reversed(sns.color_palette("hls", n)))

get_experiment_name = lambda folder_name, i: list(filter(lambda x: len(x)>2, folder_name.split('/')))[-i]


def plot_metric_curves(folders, metrics, ax=None, size_of_acquisitions=10):
    """
    Plots the curve for the metric in `metrics` for 
    each experiment found in `folders`
    """
    if not ax:
        plt.clf()
        ax = plt.gca()

    for color, folder in zip(COLORS(len(folders)), folders):
        for line_style, metric in zip(['-', '--', ':', '-.'], metrics):
            curve = np.load(os.path.join(folder, metric +'.npy'))
            ax.plot(curve, label=get_experiment_name(folder, 2)+'-'+metric, linestyle=line_style, color=color)
    ax.legend(loc=(1,0)) # put the legend outside
    xticks = ax.get_xticks()
    ax.set_xticklabels(np.array(xticks)*size_of_acquisitions)
    ax.set_xlabel('Training set size')
    sns.despine()
    return ax

def collect_replicates(global_folder, metric):
    """
    :param global_folder: this is the path to where the replicates are:
                for example if:
                PATH/bandit-ucb/
                        replicate1
                        replicate2
                        replicate3

                You will supply PATH/bandit-ucb
    :param metric: the metric that needs to be plotted

    :returns: the list of the paths that need to be loaded:
        [ PATH/bandit-ucb/replicate1/myreward.npy,
          PATH/bandit-ucb/replicate2/myreward.npy,
          PATH/bandit-ucb/replicate3/myreward.npy ]
    """
    replicates = glob(os.path.join(global_folder, '*'))

    paths = []
    for replicate in replicates:
        paths.extend(glob(os.path.join(replicate, metric + '.npy')) \
                + glob(os.path.join(replicate, metric + '.txt')))
    
    return paths

def average_replicates(replicate_paths):
    """
    Loads the npy files in replicate_paths and returns them
    """
    datas = []
    for path in replicate_paths:
        datas.append(np.load(path))

    datas = np.array(datas)
    return np.mean(datas, axis=0), np.std(datas, axis=0)

def acquisition_function_data(replicate_paths):
    datas = []
    possible_acqusition_functions = set()
    for path in replicate_paths:
        with open(path, 'r') as f:
            # list of possible acquisiton functions
            datas.append(f.readlines())
            possible_acqusition_functions.update(set(datas[-1]))
    # process the datas so it returns intergers corresponding to the indices
    return datas, possible_acqusition_functions


def plot_acq_curves(folders, ax=None):
    """
    This will plot the acqusition curves for each replicate
    separated by the experiment name
    """
    if not ax:
        plt.clf()
        ax = plt.gca()

    possible_acqusition_functions = set()
    datas = []
    for folder in folders:
        replicate_paths = collect_replicates(folder, 'acqusition_function_history')
        acq_curves, _possible_acqs = acquisition_function_data(replicate_paths)
        possible_acqusition_functions.update(_possible_acqs)
        t = np.arange(len(acq_curves[0]))
        datas.append((t, acq_curves, get_experiment_name(folder, 1)))

    possible_acqusition_functions = list(sorted(list(possible_acqusition_functions)))

    for marker, (t, data_curves, expname) in zip(['-', '--', '.', ':-'], datas):
        plottable_datas = [[ possible_acqusition_functions.index(acq) for acq in data_ ] for data_ in data_curves]
        for (i, curve),  color in zip(enumerate(plottable_datas), COLORS(len(plottable_datas))):
            ax.plot(t, curve, label=expname+'-'+str(i), linestyle=marker, c=color)
    
    ax.legend(loc=(1,0)) # put the legend outside
    ax.set_yticks(np.arange(len(possible_acqusition_functions)))
    ax.set_yticklabels(possible_acqusition_functions)
    ax.set_xticks(t)
    ax.set_ylabel('Acqusition Function Choice')
    ax.set_xlabel('# of acquisitions')
    sns.despine()
    return ax


def plot_average_curves(folders, metrics, ax=None):
    """
    Plots the average curve for the metric in `metrics` for 
    each experiment found in `folders`. 
    Note that inside each `folder`,
    there must be atleast one replicate
    """
    if not ax:
        plt.clf()
        ax = plt.gca()

    for color, folder in zip(COLORS(len(folders)), folders):
        for line_style, metric in zip(['-', '--', ':', '-.'], metrics):
            replicate_paths = collect_replicates(folder, metric)
            mean, std = average_replicates(replicate_paths)
            t = np.arange(mean.shape[0])
            ax.plot(t, mean, label=get_experiment_name(folder, 1)+'\n'+metric, linestyle=line_style, color=color)
            ax.fill_between(t, mean+std, mean-std, facecolor=color, alpha=0.1)
    ax.legend(loc=(1,0)) # put the legend outside
    ax.set_xticks(t)
    ax.set_xlabel('# of acquisitions')
    sns.despine()
    ax.set_xlim(xmin=0)
    return ax

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-f', '--folders',
        help="""folders with the experiments. 
            This must be can be follows:
            -f FOLDER1 FOLDER2

            where FOLDER1 and FOLDER2 are two experiments
            each with subfolders with replicates:
            FOLDER1/
                replicate1
                replicate2
            FOLDER2/
                replicate1
                replicate2
            """,
        nargs='+', required=True, type=str)

    named_args.add_argument('-m', '--metrics',
        help="""metrics to plot. can be multiple metrics like:
            -m val_acc train_acc
            or must be `-m acq` for plotting acquition functions""",
        nargs='+', required=True, type=str)

    named_args.add_argument('-name', '--name',
        help="""title of figure, will be saved as NAME.pdf""",
        required=True, type=str)
    
    args = parser.parse_args()

    f = plt.figure()
    ax = f.gca()
    if 'acq' in args.metrics:
        assert len(args.metrics) == 1, \
            'if plotting acqusition_function_history, it must be the only metric when running this script'
        ax = plot_acq_curves(args.folders, ax=ax)
    else:
        ax = plot_average_curves(args.folders, args.metrics, ax=ax)

    ax.set_title(args.name)
    f.savefig(args.name+'.pdf', bbox_inches='tight')
