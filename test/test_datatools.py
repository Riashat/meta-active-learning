from src import datatools
import numpy as np

get_shape = lambda tuple_: list(map(lambda x: x.shape, tuple_))

def test_data_pipeline():

	original_training_data, testing_data = datatools.get_mnist()
	assert len(original_training_data[0].shape) == 4
	
	training_data, validation_data = datatools.get_valid_data(*original_training_data)
	
	x_shape, y_shape = get_shape(training_data)
	val_x_shape, val_y_shape = get_shape(validation_data)

	assert x_shape[0] > val_x_shape[0]
	assert x_shape[1:] == val_x_shape[1:]
	
	training_data, pool_data = datatools.get_pool_data(*training_data)
	
	x_shape, y_shape = get_shape(training_data)
	pool_x_shape, pool_y_shape = get_shape(pool_data)

	assert pool_x_shape[0] > x_shape[0]

	assert get_shape(training_data)[0][0] == (np.max(original_training_data[1])+1)*2