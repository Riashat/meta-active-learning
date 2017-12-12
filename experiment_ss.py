from __future__ import print_function
import sys
import os
sys.path.append("./src")
sys.path.append("./src/ssl_vae/semi-supervised")
sys.path.append("./src/ssl_vae/bayesbench")
from src.utils import (
    get_parser,
    Logger,
    create_folder,
    RewardProcess,
    stochastic_evaluate
)
from ssl_vae.baby_ss_vae import *

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

batch_size = args.batch_size
epochs = args.epochs
acquisition_iterations = args.acquisitions
dropout_iterations = args.dropoutiterations
n_queries = args.queries
weight_constant = args.weight_decay
n_stoch_evaluations = 10
pool_subset_size = 2000 # the number of elements from the pool to run dropout sampling on

print ("Using dataset : ", args.data)
(x_train, y_train), val_data, (x_pool, y_pool), test_data = datatools.data_pipeline(valid_ratio=0.1, dataset=args.data)

n_classes = y_train.shape[1]


#(x_valid, y_valid) = val_data
# for testing purposes:
val_data = (val_data[0][:5000], val_data[1][:5000])
print('WARNING: only using 500 points for validation')
# test_data = (test_data[0][:500], test_data[1][:500])

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
deep_extractor = "resnet18"
params = {
    'batch_size':50,
    'num_workers':1,
    'lr_m1':3e-5,
    'lr_m2':1e-2,
    'epochs_m1':10,
    'epochs_m2':10,
    'dims':[512, 50, [600]],
    'verbose':True,
    'log':True,
    'dropout':0.1
}
model = SSClassifier(params, deep_extractor)
history_m1, history_m2 = model.fit(x_train,y_train,x_pool) # Replace x_pool by x_unlabeled

print(history_m1,history_m2)
"""
model = cnn(input_shape=x_train.shape[1:],
            output_classes=n_classes,
            bayesian= args.model == 'bayesian',
            train_size=x_train.shape[0],
            weight_constant=weight_constant)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=args.epochs) 
"""
# for efficiency purposes might want to remove testing on val set here

train_loss_m1 = history_m1['loss']

train_loss = history_m2[0]
train_accuracy = history_m2[1]['accuracy']

val_loss, val_accuracy = stochastic_evaluate(model, val_data, n_stoch_evaluations)
test_loss, test_accuracy = stochastic_evaluate(model, test_data, n_stoch_evaluations)

print ("Accuracy on validation set with initial training dataset")
print('Validation accuracy:', val_accuracy)
print ('Test Accuracy', test_accuracy)

logger.record_train_metrics(train_loss[-1], train_accuracy[-1])
logger.record_val_metrics(val_loss, val_accuracy)
logger.record_test_metrics(test_loss, test_accuracy)


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

    # x_train, y_train = datatools.combine_datasets(new_data_for_training, (x_train, y_train))
    x_train, y_train = datatools.combine_datasets((x_train, y_train), new_data_for_training)
    x_pool, y_pool = datatools.combine_datasets(pool_without_subset, pool_subset_updated)

    model = SSClassifier(params, deep_extractor)
    history_m1, history_m2 = model.fit(x_train,y_train,x_pool_subset) # replace last arg by x_unlabeled

    train_loss_m1 = history_m1['loss']

    train_loss = history_m2[0]
    train_accuracy = history_m2[1]['accuracy']

    # this val_accuracy is used to update policy
    val_loss, val_accuracy = stochastic_evaluate(model, val_data, n_stoch_evaluations)

    print('Validation accuracy:', val_accuracy)

    logger.record_train_metrics(train_loss[-1], train_accuracy[-1])
    logger.record_val_metrics(val_loss, val_accuracy)
    logger.record_acquisition_function(acquisition_function_name)
    # get the reward for making the acquisition
    reward = reward_process.get_reward(prev_acc,
                                       val_accuracy,
                                       prev_loss,
                                       val_loss)
    
    # update the policy based on this reward
    # note that internally the last action selected
    # is stored.
    # print('Reward gained:', reward)
    logger.record_reward(reward)

    policy.update_policy(reward, verbose=True)

    prev_loss = val_loss
    prev_acc = val_accuracy

    test_loss, test_accuracy = stochastic_evaluate(model, test_data, n_stoch_evaluations)

    print ('Test Accuracy', test_accuracy)
    logger.record_test_metrics(test_loss, test_accuracy)

logger.save()
print('DONE')
