from src.networks import cnn

def test_cnn():

	assert cnn((32, 32, 1), 10)