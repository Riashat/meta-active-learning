# import keras

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
# from keras.losses import categorical_crossentropy
# from keras.optimizers import Adam
# from keras.layers.core import Lambda
# from keras import regularizers



# from __future__ import print_function
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
from keras import backend as K  
import random
random.seed(5001)
from keras.regularizers import l2, activity_l2




# bayesian CNN
def cnn(input_shape,
        output_classes,
        bayesian=True,
        train_size=20,
        weight_constant=3):

        n_filters = 32
        # weight_constant = 3
        # size of pooling area for max pooling
        nb_pool = 2
        # convolution kernel size
        nb_conv = 3
        optimizer = 'adam'
        model = Sequential()
        model.add(Convolution2D(n_filters, nb_conv, nb_conv, border_mode='valid', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(n_filters, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))
        
        # Weight_Decay =  / float(train_size)
        print ("L2 Weight Decay", weight_constant / float(train_size))
        model.add(Flatten())
        model.add(Dense(256, W_regularizer=l2(weight_constant / float(train_size))))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_classes))
        model.add(Activation('softmax'))


        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        return model



# bayesian CNN
# def other_cnn(input_shape,
#         output_classes,
#         conv_kernel_size= (3, 3),
#         n_filters= 32,
#         pool_size=(2, 2),
#         kernel_regularizer=regularizers.l2(0.01),
#         #kernel_regularizer=None,
#         # activity_regularizer=regularizers.l1(0.01),
#         activity_regularizer=None,
#         optimizer=Adam(lr=0.001, decay=1e-6),
#         bayesian=True,
#         train_size=20,
#         weight_constant=1):

#     """
#     Returns a CNN that can be used for mnist
#     :param input_shape: the size of the input
#     :param conv_kernel_size: size of the kernel for the convolution layers
#     :param n_filters: the number of filters per convolution layer
#     :param pool_size: size of the pooling operation
#     :param kernel_regularizer: the regularizer to use for the weights of the dense layer
#     :param activity_regularizer: the regularizer to use for the activations of a network
#     :param optimizer: the optimizer to use (must be a keras.optimizers.Optimizer instance)
#     :param loss: the loss function to use 
#     :returns: keras.models.Model instance that has been compiled.
#     """
#     if bayesian:
#         print('Using bayesian network')
#     else:
#         print('Using deterministic network')
        
#     model = Sequential()
#     model.add(Conv2D(n_filters,
#                      kernel_size=conv_kernel_size,
#                      activation='relu',
#                      input_shape=input_shape))

#     """
#     TODO : Using 2x filters in the CNN model here
#     """
#     model.add(Conv2D(n_filters*2, conv_kernel_size, activation='relu'))
#     model.add(MaxPooling2D(pool_size=pool_size))
    
#     if bayesian:
#         model.add(Lambda(lambda x: K.dropout(x, level=0.25)))
#     else:
#         model.add(Dropout(0.25))

#     model.add(Flatten())
#     model.add(Dense(128,
#                     activation='relu',
#                     kernel_regularizer=regularizers.l2(weight_constant / float(train_size)),
#                     activity_regularizer=activity_regularizer))

#     if bayesian:
#         model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
#     else:
#         model.add(Dropout(0.5))

#     model.add(Dense(output_classes, activation='softmax'))
#     model.compile(loss=categorical_crossentropy,
#                   optimizer=optimizer,
#                   metrics=['accuracy'])

#     return model
