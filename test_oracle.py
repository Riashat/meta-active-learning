import numpy as np
from keras.datasets import mnist
from oracle import KNOracle
(x_train, y_train), (x_test, y_test) = mnist.load_data()


oracle = KNOracle(x_train, y_train)
print('oracle trained')
print('making predictions:')
print(np.sum(y_test[:50] == oracle.assign_nearest_available_label(x_test[:50]))/float(x_test[:50].shape[0]))

