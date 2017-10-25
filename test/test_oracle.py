import numpy as np
from keras.datasets import mnist
from oracle import KNOracle
(x_train, y_train), (x_test, y_test) = mnist.load_data()


oracle = KNOracle(x_train, y_train)
print('oracle trained')
print('making predictions:')
print(np.sum(y_test[:50] == oracle.assign_nearest_available_label(x_test[:50]))/float(x_test[:50].shape[0]))
data, labels = oracle.return_nearest_available_example_and_label(x_test[:2], neighbors=2)

print(data.shape)
print(labels)
#print(np.equal(oracle.classifier._fit_X, x_train.reshape(-1, x_train.shape[1]**2)))
