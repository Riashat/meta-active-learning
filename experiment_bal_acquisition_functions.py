'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from scipy.spatial.distance import pdist, squareform
from keras.layers.core import Lambda
import numpy as np
import scipy as sp
from scipy import linalg
from random import randint
from keras import regularizers
import random
import pdb
import argparse
import os
from oracle import KNOracle
from acquisition_function import acquisition_functions


parser = argparse.ArgumentParser()
named_args = parser.add_argument_group('named arguments')

named_args.add_argument('-g', '--gpu',
                        help="""gpu to use""",
                        required=False, type=str, default='0')

# named_args.add_argument('--acq_function',
# 			help="""acq_function for active learning : 'bald', 'random', 'varratio', 'segnet', 'maxentropy'  """,
#       default='bald', 
# 			required=True, type=str)

named_args.add_argument('-e', '--epochs',
      help="""# of epochs to train""",
      required=False, type=int, default=1000)

named_args.add_argument('--policy',
      help="""acq_function for active learning : 'random_policy'  """,
      required=True, type=str, default='random_policy')


named_args.add_argument('-a', '--acquisitions',
      help="""# of acquisitions for active learning""",
      required=False, type=int, default=980)


named_args.add_argument('-t', '--n_experiments',
      help="""# of experiments with different random seeds""",
      required=False, type=int, default=5)

named_args.add_argument('-d', '--dropoutiterations',
      help="""# of dropout estimates""",
      required=False, type=int, default=100)

named_args.add_argument('-s', '--save_dir',
      help="""`""",
      required=True, type=str, default='./results/')




args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def get_valid_data(x_train, y_train):
    #split into train and validation sets
    valid_shape = np.int(np.round(x_train.shape[0]*0.1))
    x_valid = x_train[-valid_shape:, :, :, :]
    y_valid = y_train[-valid_shape:]

    x_train = x_train[0:-valid_shape, :, :, :]
    y_train = y_train[0:-valid_shape]


    return x_train, y_train, x_valid, y_valid


def get_pool_data(x_train, y_train, initial_train_point):
    x_pool = x_train[initial_train_point:, :, :, :]
    y_pool = y_train[initial_train_point:]

    x_train = x_train[0:initial_train_point, :, :, :]
    y_train = y_train[0:initial_train_point]

    return x_train, y_train, x_pool, y_pool

def get_uniform_training_data(x_train, y_train):

  idx_0 = np.array(  np.where(y_train == 0) ).T
  idx_0 = idx_0[0:2, 0]
  x_0 = x_train[idx_0, :, :, :]
  y_0 = y_train[idx_0]

  idx_1 = np.array(  np.where(y_train == 1) ).T
  idx_1 = idx_1[0:2, 0]
  x_1 = x_train[idx_1, :, :, :]
  y_1 = y_train[idx_1]

  idx_2 = np.array(  np.where(y_train == 2) ).T
  idx_2 = idx_2[0:2, 0]
  x_2 = x_train[idx_2, :, :, :]
  y_2 = y_train[idx_2]

  idx_3 = np.array(  np.where(y_train == 3) ).T
  idx_3 = idx_3[0:2, 0]
  x_3 = x_train[idx_3, :, :, :]
  y_3 = y_train[idx_3]

  idx_4 = np.array(  np.where(y_train == 4) ).T
  idx_4 = idx_4[0:2, 0]
  x_4 = x_train[idx_4, :, :, :]
  y_4 = y_train[idx_4]

  idx_5 = np.array(  np.where(y_train == 5) ).T
  idx_5 = idx_5[0:2, 0]
  x_5 = x_train[idx_5, :, :, :]
  y_5 = y_train[idx_5]

  idx_6 = np.array(  np.where(y_train == 6) ).T
  idx_6 = idx_6[0:2, 0]
  x_6 = x_train[idx_6, :, :, :]
  y_6 = y_train[idx_6]

  idx_7 = np.array(  np.where(y_train == 7) ).T
  idx_7 = idx_7[0:2, 0]
  x_7 = x_train[idx_7, :, :, :]
  y_7 = y_train[idx_7]

  idx_8 = np.array(  np.where(y_train == 8) ).T
  idx_8 = idx_8[0:2, 0]
  x_8 = x_train[idx_8, :, :, :]
  y_8 = y_train[idx_8]

  idx_9 = np.array(  np.where(y_train == 9) ).T
  idx_9 = idx_9[0:2, 0]
  x_9 = x_train[idx_9, :, :, :]
  y_9 = y_train[idx_9]


  x_train = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis=0)
  y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0)


  return x_train, y_train


batch_size = 128
num_classes = 10
epochs = args.epochs
acquisition_iterations = args.acquisitions
## use MC Samples = 100
dropout_iterations = args.dropoutiterations
Queries = 10


# input image dimensions
img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 4
nb_conv = 3

Experiments = args.n_experiments

Exp_Training_Loss = np.zeros(shape=(acquisition_iterations+1, Experiments))
Exp_Valid_Acc = np.zeros(shape=(acquisition_iterations+1, Experiments))
Exp_Test_Acc = np.zeros(shape=(acquisition_iterations+1, Experiments))
 

for e in range(Experiments):

  print ("EXPERIMENT NUMBER", e)

  # the data, shuffled and split between train and test sets
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  if K.image_data_format() == 'channels_first':
      print ("Using Channels first")
      x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      input_shape = (1, img_rows, img_cols)
  else:
      print("Channels last")
      x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)


  x_train, y_train, x_valid, y_valid = get_valid_data(x_train, y_train)

  initial_train_point = 10000 
  x_train, y_train, x_pool, y_pool = get_pool_data(x_train, y_train, initial_train_point)

  x_train, y_train = get_uniform_training_data(x_train, y_train)

  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print (x_pool.shape[0], 'pool samples')
  print(x_valid.shape[0], 'valid samples')
  print(x_test.shape[0], 'test samples')

  ### normalize and convert to categorical
  x_train = x_train.astype('float32')
  x_valid = x_valid.astype('float32')
  x_pool = x_pool.astype('float32')
  x_test = x_test.astype('float32')

  x_train /= 255
  x_valid /= 255
  x_pool /= 255
  x_test /= 255
  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_valid = keras.utils.to_categorical(y_valid, num_classes)
  # y_pool = keras.utils.to_categorical(y_pool, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  all_train_loss = np.zeros(shape=(acquisition_iterations+1, 1))
  all_valid_acc = np.zeros(shape=(acquisition_iterations+1, 1))

  all_test_acc = np.zeros(shape=(acquisition_iterations+1, 1))

  model = Sequential()
  model.add(Conv2D(nb_filters, kernel_size=(nb_conv, nb_conv),
                   activation='relu',
                   input_shape=input_shape))
  model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Lambda(lambda x: K.dropout(x, level=0.25)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
  model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
  model.add(Dense(num_classes, activation='softmax'))
  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])


  """
  For testing purposes
  """
  x_valid = x_valid[0:500, :, :, :]
  y_valid = y_valid[0:500, :]
  x_test = x_test[0:500, :, :, :]
  y_test = y_test[0:500, :]

  hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_valid, y_valid))

  train_loss = np.asarray(hist.history.get('loss'))
  valid_acc = np.asarray(hist.history.get('val_acc'))

  all_train_loss[0,:] = train_loss[-1]
  all_valid_acc[0,:] = valid_acc[-1]


  score, accuracy = model.evaluate(x_test, y_test, verbose=0)
  print ("Accuracy with initial dataset")
  print('Test accuracy:', accuracy)

  all_test_acc[0,:] = accuracy

  policy = args.policy



  for i in range(acquisition_iterations):

    print ("Acquisition Iteration", i)

    #take subset of pool points - to evaluate uncertainty over these points for querying
    pool_subset = 2000
    pool_subset_dropout = np.asarray(random.sample(range(0,x_pool.shape[0]), pool_subset))
    X_Pool_Dropout = x_pool[pool_subset_dropout, :, :, :]
    y_Pool_Dropout = y_pool[pool_subset_dropout]

    """
    Baseline for BANDITS APPROACH : Select one of the acquisition functions based on a "random policy"
    """
    if policy == "random_policy":
      all_acuisition_functions = ['bald', 'maxentropy', 'varratio', 'segnet']
      acq_function = random.choice(all_acuisition_functions)
    else:
      raise Exception('Need a valid acquisition function')


    print ("Using acquisition function :", acq_function)
    uncertain_pool_points = acquisition_functions(acq_function, X_Pool_Dropout, num_classes, model, batch_size, dropout_iterations)

    a_1d = uncertain_pool_points.flatten()
    x_pool_index = a_1d.argsort()[-Queries:][::-1]

    Pooled_X = X_Pool_Dropout[x_pool_index, :, :, :]
    Pooled_Y = y_Pool_Dropout[x_pool_index] 

    ## convert y_pool to categorical here
    Pooled_Y = keras.utils.to_categorical(Pooled_Y, num_classes)  

    delete_Pool_X = np.delete(x_pool, (pool_subset_dropout), axis=0)
    delete_Pool_Y = np.delete(y_pool, (pool_subset_dropout), axis=0)
    delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (x_pool_index), axis=0)
    delete_Pool_Y_Dropout = np.delete(y_Pool_Dropout, (x_pool_index), axis=0)

    x_pool = np.concatenate((x_pool, X_Pool_Dropout), axis=0)
    y_pool = np.concatenate((y_pool, y_Pool_Dropout), axis=0)

    x_train = np.concatenate((x_train, Pooled_X), axis=0)
    y_train = np.concatenate((y_train, Pooled_Y), axis=0)

    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size=(nb_conv, nb_conv),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Lambda(lambda x: K.dropout(x, level=0.25)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid))

    train_loss = np.asarray(hist.history.get('loss'))
    valid_acc = np.asarray(hist.history.get('val_acc'))

    all_train_loss[i,:] = train_loss[-1]
    all_valid_acc[i,:] = valid_acc[-1]

    score, accuracy = model.evaluate(x_test, y_test, verbose=0)

    all_test_acc[i,:] = accuracy


  np.save(args.save_dir + acq_function + "_experiment_" + str(e) +  "_train_loss.npy", all_train_loss)
  np.save(args.save_dir + acq_function + "_experiment_" + str(e) +  "_valid_acc.npy", all_valid_acc)
  np.save(args.save_dir + acq_function + "_experiment_" + str(e) +  "_test_acc.npy", all_test_acc)

  Exp_Training_Loss[:, e] = all_train_loss[:,0]
  Exp_Valid_Acc[:, e] = all_valid_acc[:,0]
  Exp_Test_Acc[:, e] = all_test_acc[:, 0]



np.save(args.save_dir + acq_function + "_all_train_loss.npy", Exp_Training_Loss)
np.save(args.save_dir + acq_function + "_all_valid_acc.npy", Exp_Valid_Acc)
np.save(args.save_dir + acq_function + "_all_test_acc.npy", Exp_Test_Acc)

