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
import numpy as np
import scipy as sp
from scipy import linalg
import random
import pdb


def split_data_into_binary(x_train, y_train, x_test, y_test):
    class_2 = np.where(y_train==2)[0]
    class_8 = np.where(y_train==8)[0]
    x_2 = x_train[class_2, :, :, :]
    y_2 = y_train[class_2]
    x_8 = x_train[class_8, :, :, :]
    y_8 = y_train[class_8]

    x_train = np.concatenate((x_2, x_8), axis=0)
    y_train = np.concatenate((y_2, y_8), axis=0)

    class_2 = np.where(y_test==2)[0]
    class_8 = np.where(y_test==8)[0]

    x_2 = x_test[class_2, :, :, :]
    y_2 = y_test[class_2]
    x_8 = x_test[class_8, :, :, :]
    y_8 = y_test[class_8]

    x_test = np.concatenate((x_2, x_8), axis=0)
    y_test = np.concatenate((y_2, y_8), axis=0)

    #randomly shuffle the data again
    random_split = np.asarray(random.sample(range(0,x_train.shape[0]), x_train.shape[0]))
    x_train = x_train[random_split, :, :, :]
    y_train = y_train[random_split]

    random_split = np.asarray(random.sample(range(0,x_test.shape[0]), x_test.shape[0]))
    x_test = x_test[random_split, :, :, :]
    y_test = y_test[random_split]


    return x_train, y_train, x_test, y_test


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



batch_size = 128
num_classes = 10
epochs = 4

# input image dimensions
img_rows, img_cols = 28, 28

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




# x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
# x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
# input_shape = (1, img_rows, img_cols)

x_train, y_train, x_test, y_test = split_data_into_binary(x_train, y_train, x_test, y_test)
x_train, y_train, x_valid, y_valid = get_valid_data(x_train, y_train)


initial_train_point = 10
x_train, y_train, x_pool, y_pool = get_pool_data(x_train, y_train, initial_train_point)


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
y_pool = keras.utils.to_categorical(y_pool, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print ("Accuracy with initial dataset")
print('Test accuracy:', score[1])


print ("Performing Active Learning with MBR")
acquisition_iterations = 10

for i in range(acquisition_iterations):

  print ("Acquisition Iteration", i)

  #compute the RBF kernel - similarity between labeled and unlabeled data points

  """
  Can we take a subset of pool points here?
  """
  ##TO DO (CHECK!)
  x_pool = x_pool[0:1000, :, :, :]
  all_data = np.concatenate((x_train, x_pool), axis=0)
  all_data = all_data.reshape(all_data.shape[0], img_rows*img_cols)

  sigma  = 1
  pairwise_dists = squareform(pdist(all_data, 'euclidean'))
  W = sp.exp(pairwise_dists ** 2 / sigma ** 2)


  #compute the combinatorial Laplacian
  d_i = W.sum(axis=1)
  D = np.diag(d_i)

  Delta = D - W 

  #computing the harmonic function - without any acquisitions yet
  Delta_ll = Delta[0:x_train.shape[0], 0:x_train.shape[0]]
  Delta_ul = Delta[x_train.shape[0]:, 0:x_train.shape[0]]
  Delta_lu = Delta[0:x_train.shape[0], x_train.shape[0]:]
  Delta_uu = Delta[x_train.shape[0]:, x_train.shape[0]:]


  inv_Delta_uu = linalg.inv(Delta_uu)
  All_f_L = y_train
  Delta_mult = np.dot(inv_Delta_uu, Delta_ul)
  All_f_U = - np.dot(Delta_mult, All_f_L)   


  """
  We can consider computing harmonic function over subset of pool points?
  """
  #f_I is the entire harmonic function over all the data points (U + L)
  All_f_I = np.concatenate((All_f_L, All_f_U), axis=0)

  pool_subset = 50

  print('Compute Expected Bayes Risk for ALL Pool Points in Acquisition Iteration ', i)
  Bayes_Risk = np.zeros(shape=(x_pool.shape[0]))

  for k in range(pool_subset):
      pool_point = x_pool[np.array([k]), :, :, :]
      pool_point_y = y_pool[np.array([k])]

      #add this pool point to train data
      X_train_Temp = np.concatenate((x_train, pool_point), axis=0)
      y_train_Temp = np.concatenate((y_train, pool_point_y), axis=0)

      ### TO DO : check
      #Should we delete this pool point from pool set
      # X_Pool_Temp = np.delete(x_pool, k, 0)

      ## W and D stays the same - only Delta_uu, Delta_ul etc changes
      Delta_ll = Delta[0:X_train_Temp.shape[0], 0:X_train_Temp.shape[0]]
      Delta_ul = Delta[X_train_Temp.shape[0]:, 0:X_train_Temp.shape[0]]
      Delta_lu = Delta[0:X_train_Temp.shape[0], X_train_Temp.shape[0]:]
      Delta_uu = Delta[X_train_Temp.shape[0]:, X_train_Temp.shape[0]:]

      #compute the new changed f
      inv_Delta_uu = linalg.inv(Delta_uu)
      f_L = y_train_Temp
      Delta_mult = np.dot(inv_Delta_uu, Delta_ul)
      f_U = - np.dot(Delta_mult, f_L)     
      f_I = np.concatenate((f_L, f_U), axis=0)

      #compute the new estimated Bayes risk for this added point
      R = np.array([0])
      for m in range(f_I.shape[0]):
        val_f_I = f_I[m]
        other_val_f_I = 1 - val_f_I
        min_val = np.amin(np.array([val_f_I, other_val_f_I]))
        R = R + min_val
      Estimated_Risk = R

      #we need f_k values for each pool point in consideration
      f_All_Pool = All_f_I[All_f_L.shape[0]:]
      f_k = f_All_Pool[k]

      pdb.set_trace()

      Bayes_Risk = (1 - f_k) * Estimated_Risk + (f_k)*Estimated_Risk

  print('Finished Computing Bayes Risk for Unlabelled Pool Points')


  pdb.set_trace()


