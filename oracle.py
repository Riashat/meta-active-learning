"""
An oracle that can return labels of 
points and hypothesize labels of points
it has never seen
"""

from sklearn.neighbors import KDTree, KNeighborsClassifier

def convert_2d_to_1d(X):
	return X.reshape(-1, X.shape[1]*X.shape[2])

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