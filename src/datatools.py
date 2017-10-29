import keras
from keras import backend as K
import numpy as np
import random

def get_mnist():
    """
    Returns the MNIST dataset formattted and ready for training
    """
    from keras.datasets import mnist

    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10
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

    return (x_train, y_train), (x_test, y_test)

def get_cifar10():
    """
    Returns the CIFAR 10 dataset formattted and ready for training
    """

    from keras.datasets import cifar10

    # input image dimensions
    img_rows, img_cols = 32, 32
    n_channels = 3
    num_classes = 10
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if K.image_data_format() == 'channels_first':
        print ("Using Channels first")
        x_train = x_train.reshape(x_train.shape[0], n_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], n_channels, img_rows, img_cols)
        input_shape = (n_channels, img_rows, img_cols)
    else:
        print("Channels last")
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
        input_shape = (img_rows, img_cols, n_channels)

    return (x_train, y_train), (x_test, y_test)

def prep(x, y):
    x = x.astype('float32')
    x /= 255
    n_classes = 10
    y = keras.utils.to_categorical(y, n_classes)
    return x, y


def get_valid_data(x_train, y_train, valid_ratio=0.1):
    """
    split into train and validation sets
    :param x_train: the x input
    :param y_train: the y_input
    :param valid_ratio: the percentage of data to include in the validation set
    """

    assert x_train.shape[0] == y_train.shape[0]

    valid_shape = np.int(np.round(x_train.shape[0]*valid_ratio))
    x_valid = x_train[-valid_shape:]
    y_valid = y_train[-valid_shape:]

    x_train = x_train[:-valid_shape]
    y_train = y_train[:-valid_shape]

    return (x_train, y_train), (x_valid, y_valid)


def get_pool_data(x, y):
    """
    Creates a pool set
    :param x: the training data inputs
    :param y: the training data labels
    :param initial_train_point: the starting position to start selecting
    """

    y_classes = np.max(y)+1

    xs = [] # xs to put in the training set
    ys = [] # ys to put in the training set
    idxs = [] # indexes of data put in the training set
    for y_class in range(y_classes):
        idx = np.array(  np.where(y == y_class) ).T
        idx = idx[0:2, 0]
        xs.append(x[idx])
        ys.append(y[idx])
        idxs.extend(idx)

    x_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)

    x_pool = np.delete(x, idxs, axis=0)
    y_pool = np.delete(y, idxs, axis=0)
    
    return (x_train, y_train), (x_pool, y_pool)

def data_pipeline(valid_ratio=0.1, dataset='mnist'):

    # get training and testing data
    if dataset == 'mnist':
        training_data, testing_data = get_mnist()
    elif dataset == 'cifar10':
        training_data, testing_data = get_cifar10()
    else:
        raise ValueError('No dataset found!')
    # get validation data
    training_data, validation_data = get_valid_data(*training_data, valid_ratio=valid_ratio)
    
    # get pool data
    training_data, pool_data = get_pool_data(*training_data)
    
    # normalize x's and convert y's to categorical arrays
    training_data = prep(*training_data)    
    testing_data = prep(*testing_data)
    validation_data = prep(*validation_data)
    pool_data = prep(*pool_data)


    return training_data, validation_data, pool_data, testing_data

def get_pool_subset(X_pool, Y_pool, subset_size=2000):
    """
    Creates a subset of the pool data for 
    running dropout and getting uncertainties
    :param X_pool: the complete pool data
    :param Y_pool: the complete pool labels
    :param subset_size:the size of the subset
    :return: (X_pool_prime, Y_pool_prime) the original pool set with the subset removed.
    :return: (X_pool_subset, Y_pool_subset) a subset of `sub
    set_size` points from X_pool
    """
    subset_indices = np.asarray(random.sample(range(0,X_pool.shape[0]), subset_size))
    X_pool_subset = X_pool[subset_indices]
    Y_pool_subset = Y_pool[subset_indices]
    X_pool = np.delete(X_pool, subset_indices, axis=0)
    Y_pool = np.delete(Y_pool, subset_indices, axis=0)
    return (X_pool, Y_pool), (X_pool_subset, Y_pool_subset)

def combine_datasets(dataset_1, dataset_2):
    x1, y1 = dataset_1
    x2, y2 = dataset_2

    return np.concatenate((x1, x2), axis=0), np.concatenate((y1, y2), axis=0)
