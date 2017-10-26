from src.networks import cnn
from src import datatools
import numpy as np

def test_cnn():

	assert cnn((32, 32, 1), 10)


def test_bayesian_training():
    # use the third return because it has more points
    # _, (x_val, y_val), (x_train, y_train), _ = data_pipeline(valid_ratio=0.1)
    train_data, val_data = datatools.get_mnist()
    x_train, y_train = datatools.prep(*train_data)
    x_val, y_val = datatools.prep(*val_data)

    # (x_train_1, y_train_1), (x_test_1, y_test_1) = data_is_correct()
    # assert np.all(x_train_1 == x_train)
    # assert np.all(y_train_1 == y_train)

    net = cnn((28, 28, 1), 10, bayesian=True)

    print('training')
    net.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1)

    loss, acc = net.evaluate(x_val, y_val)

    assert acc > 0.80

def test_regular_training():
    # use the third return because it has more points
    _, (x_val, y_val), (x_train, y_train), _ = datatools.data_pipeline(valid_ratio=0.1)

    net = cnn((28, 28, 1), 10, bayesian=False)

    print('training')
    net.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1)

    loss, acc = net.evaluate(x_val, y_val)

    assert acc > 0.80
