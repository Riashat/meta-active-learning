from src.networks import bayesian_cnn

def test_cnn():

	assert bayesian_cnn((32, 32, 1), 10)