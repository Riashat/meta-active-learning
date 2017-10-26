from __future__ import print_function
import os
from src.utils import get_parser, Logger, create_folder

args = get_parser().parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import tensorflow as tf

# Setting a seed as described in https://github.com/blei-lab/edward/pull/184
# this is is useful for reproducibility
# !!! DO NOT MOVE THE SEED TO AFTER IMPORTING KERAS !!!

np.random.seed(args.seed)
tf.set_random_seed(args.seed)
if len(tf.get_default_graph()._nodes_by_id.keys()) > 0:
    raise RuntimeError('Seeding is not supported after initializing a part ' +
                       'of the graph.')

from src import datatools
from src.networks import cnn
from src.oracle import ask_oracle
from src.acquisition_function import run_acquisition_function, ACQUISITION_FUNCTIONS_TEXT
from src.policies import policy_parser

batch_size = 128
num_classes = 10
epochs = args.epochs
acquisition_iterations = args.acquisitions
dropout_iterations = args.dropoutiterations
n_queries = 10
pool_subset_size = 2000 # the number of elements from the pool to run dropout sampling on


(x_train, y_train), val_data, (x_pool, y_pool), test_data = datatools.data_pipeline(valid_ratio=0.1)
n_classes = y_train.shape[1]

# for testing purposes:
val_data = (val_data[0][:500], val_data[1][:500])
print('WARNING: only using 500 points for validation')


print('POLICY: ',args.policy)

# this is the policy by which one should choose acquisition functions
policy = policy_parser(args.policy, args)

# this is the reward that is calculated based on previous acc/val 
# and current acc/val 
reward_process = RewardProcess(args.reward)

# logger to record experiments
logger = Logger(experiment_name=args.policy, folder=args.folder)
logger.save_args(args)
print('Saving to ', logger.save_folder)

print('Starting Experiment')


"""
GET INITIAL ESTIMATE OF VALIDATION ACCURACY
"""
model = cnn(input_shape=x_train.shape[1:],
            output_classes=n_classes,
            bayesian= args.model == 'bayesian')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=args.epochs,
                    validation_data=val_data) 
# for efficiency purposes might want to remove testing on val set here

train_loss = history.history.get('loss')
train_accuracy = history.history.get('acc')


val_loss, val_accuracy = model.evaluate(*val_data, verbose=0)

print ("Accuracy on validation set with initial training dataset")
print('Validation accuracy:', val_accuracy)

logger.record_train_metrics(train_loss[-1], train_accuracy[-1])
logger.record_val_metrics(val_loss, val_accuracy)

prev_loss = val_loss
prev_acc = val_accuracy

"""
START COLLECTING A NEW DATASET
"""

for i in range(acquisition_iterations):
    print('Acquisition iteration: ', i)

    # get a subset of the pool:
    pool_without_subset, (x_pool_subset, y_pool_subset) = datatools.get_pool_subset(x_pool, y_pool, pool_subset_size)
    
    # get the acquisition function to use:
    acquisition_function_name = policy.get_acquisition_function()

    # get the uncertainty estimates for the pool subset
    uncertainty_estimates = run_acquisition_function(acquisition_function_name,
                                                     x_pool_subset,
                                                     n_classes,
                                                     model,
                                                     dropout_iterations=dropout_iterations)
    
    # ask the oracle for labels of the top 'n_queries' uncertain points
    new_data_for_training, pool_subset_updated = ask_oracle(uncertainty_estimates,
                                                            n_queries,
                                                            x_pool_subset,
                                                            y_pool_subset,
                                                            n_classes=n_classes)

    x_train, y_train = datatools.combine_datasets(new_data_for_training, (x_train, y_train))
    x_pool, y_pool = datatools.combine_datasets(pool_without_subset, pool_subset_updated)


    model = cnn(input_shape=x_train.shape[1:],
                output_classes=n_classes,
                bayesian= args.model == 'bayesian')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=args.epochs,
                        validation_data=val_data)


    train_loss = history.history.get('loss')
    train_accuracy = history.history.get('acc')

    # this val_accuracy is propa
    val_loss, val_accuracy = model.evaluate(*val_data, verbose=0)

    print ("Accuracy on validation set after the ", i, 'th acquisition')
    print('Validation accuracy:', val_accuracy)

    logger.record_train_metrics(train_loss[-1], train_accuracy[-1])
    logger.record_val_metrics(val_loss, val_accuracy)
    

    # get the reward for making the acquisition
    reward = reward_process.get_reward(prev_acc,
                                       val_accuracy,
                                       prev_loss,
                                       val_loss)
    
    # update the policy based on this reward
    # note that internally the last action selected
    # is stored.
    print('Reward gained:', reward)
    policy.update_policy(reward, verbose=True)

    prev_loss = val_loss
    prev_acc = val_accuracy

logger.save()
print('DONE')