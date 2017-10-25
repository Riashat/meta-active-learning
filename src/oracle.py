"""
An oracle that can return labels of 
points and hypothesize labels of points
it has never seen
"""

from sklearn.neighbors import KDTree, KNeighborsClassifier

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
