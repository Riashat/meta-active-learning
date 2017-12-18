from __future__ import print_function
import sys
import os
sys.path.append("./src")
sys.path.append("./src/ssl_vae/probtorch")
#sys.path.append("./src/ssl_vae/semi-supervised")
#sys.path.append("./src/ssl_vae/bayesbench")
from src.utils import (
    get_parser,
    Logger,
    create_folder,
    RewardProcess,
    stochastic_evaluate
)
import keras
from ssl_vae.baby_ss_vae import *

args = get_parser().parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import tensorflow as tf
import pickle
import torch

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

def reshape_train_pool(x_train,y_train,x_pool,y_pool,train_size):
    x,y = datatools.combine_datasets((x_train,y_train),(x_pool,y_pool))
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:train_size], indices[train_size:]
    return (x[training_idx,:], y[training_idx,:]), (x[test_idx,:], y[test_idx,:])

#(x_valid, y_valid) = val_data
# for testing purposes:
#val_data = (val_data[0][:5000], val_data[1][:5000])
#print('WARNING: only using 500 points for validation')
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
logger_file = open(logger.save_folder+"/log.log","w")
params = {
        'classes':10 if args.data in ['mnist','cifar10'] else y_train.shape[-1],
        'batch_size':args.batch_size,
        'lr':args.lr,
        'epochs':args.epochs,
        'dims':[784, 50, [600]], # 784: MNIST, 512: CIFAR10
        'samples':args.samples,
        'beta1':0.9,
        'beta2':0.999,
        'eps':1e-9,
        'cuda':torch.cuda.is_available(),
        'logger':logger_file,
        'cnn':args.cnn
}

"""
parameters for CNN
"""
aux_params = {
    'n_channels':1,
    'img_rows':28,
    'img_cols':28
}

train_size = args.training_size
pool_size = args.pool_size

if args.sanity_check == 1: # Make sure the network's accuracy is improving with # of labeled examples
    train_sizes = [50,150,500,1000,3000,5000]
    pool_size = 5000
    sizes = {}
    for train_size in train_sizes:
        (x_train,y_train) , (x_pool,y_pool) = reshape_train_pool(x_train,y_train,x_pool,y_pool,train_size)
        x_pool = x_pool[:pool_size]
        filepath = open(logger.save_folder+"/train_sizes.pickle","wb")
        model = SSClassifier(params,**aux_params)
        history = model.fit(x_train,y_train,x_pool)
        val_loss ,val_accuracy = model.evaluate(val_data[0],val_data[1])
        test_loss ,test_accuracy = model.evaluate(test_data[0],test_data[1])
        train_loss = np.max(np.mean(history[:,0:2],axis=1),axis=0) # Take average labelled and unlabelled elbo and take max over epochs
        train_accuracy = np.max(history[:,2],axis=0)
        sizes[train_size]=history
        pickle.dump(sizes,filepath)
    filepath.close()
    logger_file.close()
    print("Sanity check 1 done")
    exit()
elif args.sanity_check == 2: # Check whether predictions are stochastic
    (x_train,y_train) , (x_pool,y_pool) = reshape_train_pool(x_train,y_train,x_pool,y_pool,train_size)
    x_pool = x_pool[:pool_size]
    model = SSClassifier(params,**aux_params)
    history = model.fit(x_train,y_train,x_pool)
    preds = []
    trials = 5
    for trial in range(trials):
        _, pred = model.predict(val_data[0])
        preds.append(pred.numpy())
    preds = np.array(preds)
    print("Element-wise predictive variances: ")
    print(np.var(preds,axis=0))
    print("Sanity check 2 done")
    exit()

(x_train,y_train) , (x_pool,y_pool) = reshape_train_pool(x_train,y_train,x_pool,y_pool,train_size)
x_pool = x_pool[:pool_size]

model = SSClassifier(params,**aux_params)
history = model.fit(x_train,y_train,x_pool)
val_loss ,val_accuracy = model.evaluate(val_data[0],val_data[1])
test_loss ,test_accuracy = model.evaluate(test_data[0],test_data[1])
train_loss = np.max(np.mean(history[:,0:2],axis=1),axis=0) # Take average labelled and unlabelled elbo and take max over epochs
train_accuracy = np.max(history[:,2],axis=0)

# for efficiency purposes might want to remove testing on val set here

print ("Accuracy on validation set with initial training dataset")
print('Validation accuracy:', val_accuracy)
print ('Test Accuracy', test_accuracy)

logger.record_train_metrics(train_loss, train_accuracy)
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
                                                     params['classes'],
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

    model = SSClassifier(params,**aux_params)
    history = model.fit(x_train,y_train,x_pool)
    val_loss ,val_accuracy = model.evaluate(val_data[0],val_data[1])
    test_loss ,test_accuracy = model.evaluate(test_data[0],test_data[1])
    train_loss = np.max(np.mean(history[:,0:2],axis=1),axis=0) # Take average labelled and unlabelled elbo and take max over epochs
    train_accuracy = np.max(history[:,2],axis=0)

    print('Validation accuracy:', val_accuracy)

    logger.record_train_metrics(train_loss, train_accuracy)
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

    print ('Test Accuracy', test_accuracy)
    logger.record_test_metrics(test_loss, test_accuracy)

logger_file.close()
logger.save()
print('DONE')
