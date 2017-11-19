"""
An oracle that can return labels of 
points and hypothesize labels of points
it has never seen
"""

from sklearn.neighbors import KDTree, KNeighborsClassifier
import keras
import numpy as np
import sys
sys.path.append('ssl_vae')

from ssl_vae import *


def convert_2d_to_1d(X):
    return X.reshape(-1, X.shape[1]*X.shape[2])
def convert_1d_to_2d(X, img_dim):
    return X.reshape(-1, img_dim, img_dim, 1)

class KNOracle(object):
    def __init__(self, X_pool_known, Y_pool_known, n_neighbors=5, n_jobs=1):
        """
        Implements an Nearest Neighbour Oracle
        :param X_pool_known: the set of x training data
        :param Y_pool_known: the set of labels for the training data
        :param n_neighbours: the number of neighbours to take points over
        :param n_jobs: number of jobs for parallelization  
        """
        classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
        classifier.fit(convert_2d_to_1d(X_pool_known), Y_pool_known)
        self.classifier = classifier

    def assign_nearest_available_label(self, X):
        """
        Returns the nearest neighbour label for the data provided
        :param X: the data to assign labels to
        :return: the Y's assigned to X
        """

        return self.classifier.predict(convert_2d_to_1d(X))

    def return_nearest_available_example_and_label(self, X, neighbors=1):
        """
        Returns the most similar example for the data provided
        where we know the label for sure.
        :param X: the data to return examples for
        :return: X's and Y's that are similar to X
        """
        close_data_indices = self.classifier.kneighbors(convert_2d_to_1d(X), return_distance=False, n_neighbors=neighbors)
        close_data_indices = close_data_indices.reshape(-1)
        close_data = self.classifier._fit_X[close_data_indices]
        close_labels = self.classifier._y[close_data_indices]
        return convert_1d_to_2d(close_data, X.shape[1] ), close_labels

"""
ssl_vae(X_labeled,Y_labeled,X_unlabeled)

@return: Y_hat_unlabeled
"""

class SSOracle(object):
    def __init__(self, X_labeled, Y_labeled, X_unlabeled):
        """
        Initialize a semi-supervised VAE
        Train it on L+U data
        """
            self.ss_VAE =ssl_vae(X_labeled,Y_labeled,X_unlabeled)
            self.ss_VAE.train()

    def assign_best_label(self,X):
        """
        After training is performed, return the estimated labels
        """
        return self.ss_VAE.predict(X).numpy()

def ask_oracle(pool_uncertainties, n_queries, X_pool, Y_pool, n_classes=10):
    """
    An oracle that reveals the labels of 
    the n_queries most uncertain points
    :param pool_uncertainties: the uncertainty for each point
    :param n_queries: number of points to ask for
    :param X_pool: the training data for which the uncertainties were estimated
    :param Y_pool: the training labels for which the uncertainties were estimated
    :return: (X_revealed, Y_revealed) the n_queries most uncertain points with labels revealed
    :return: (X_pool_prime, Y_pool_prime) the pool with revealed points removed.
    """
    pool_uncertainties = pool_uncertainties.flatten()
    # these points need to be revealed to the learner
    pool_to_be_revealed = pool_uncertainties.argsort()[-n_queries:][::-1]

    X_revealed = X_pool[pool_to_be_revealed]
    Y_revealed = Y_pool[pool_to_be_revealed] 

    
    # the new pool with the revealed points deleted
    X_pool_prime = np.delete(X_pool, pool_to_be_revealed, axis=0)
    Y_pool_prime = np.delete(Y_pool, pool_to_be_revealed, axis=0)

    return (X_revealed, Y_revealed), (X_pool_prime, Y_pool_prime)