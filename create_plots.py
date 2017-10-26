import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

get_experiment_name = lambda folder_name: list(filter(lambda x: len(x)>2, folder_name.split('/')))[-1]


def plot_metric_curves(folders, metrics, ax=None, size_of_acquisitions=10):
    if not ax:
        plt.clf()
        ax = plt.gca()


    for folder in folders:
        for metric in metrics:
            curve = np.load(os.path.join(folder, metric +'.npy'))
            ax.plot(curve, label=get_experiment_name(folder)+'-'+metric)
    ax.legend(loc='best')
    xticks = ax.get_xticks()
    ax.set_xticklabels(np.array(xticks)*size_of_acquisitions)
    ax.set_xlabel('Training set size')

    return ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-f', '--folders',
        help="""folders with the experiments""",
        nargs='+', required=True, type=str)

    named_args.add_argument('-m', '--metrics',
        help="""metrics to plot""",
        nargs='+', required=True, type=str)

    named_args.add_argument('-name', '--name',
        help="""name of the figure""",
        required=True, type=str)

    f = plt.figure()
    ax = f.gca()

    args = parser.parse_args()
    ax = plot_metric_curves(args.folders, args.metrics, ax=ax)
    f.savefig(name+'.pdf')