from src import datatools
from src.networks import cnn
from src.oracle import ask_oracle
from src.acquisition_function import maxentropy

def test_acquisition():

    (x_train, y_train), val_data, (x_pool, y_pool), test_data = datatools.data_pipeline()
    n_classes = y_train.shape[1]

    model = cnn(input_shape=x_train.shape[1:],
                output_classes=n_classes)



    pool_without_subset, (x_pool_subset, y_pool_subset) = datatools.get_pool_subset(x_pool, y_pool, 3000)

    assert pool_without_subset[0].shape[0] == x_pool.shape[0]-3000
    assert x_pool_subset.shape[0] == 3000

    uncertainty_estimates = maxentropy(x_pool_subset, n_classes, model, dropout_iterations=2)

    new_data_for_training, pool_subset_updated = ask_oracle(uncertainty_estimates, 1000, x_pool_subset, y_pool_subset, n_classes=n_classes)
    assert new_data_for_training[0].shape[0] == new_data_for_training[1].shape[0]
    x_train_new, y_train_new = datatools.combine_datasets(new_data_for_training, (x_train, y_train))
    x_pool_new, y_pool_new = datatools.combine_datasets(pool_without_subset, pool_subset_updated)

    assert x_train_new.shape[0] == x_train.shape[0] + 1000, 'Training data increases by 1000 points only'
    assert x_pool_new.shape[0] == x_pool.shape[0] - 1000, 'Pool must reduce by 1000 points only'

    # train and validation on pool just to ensure the data characteristics are unchanged.
    print('train new shape:', x_train_new.shape)
    print('pool new shape:', x_pool_new.shape)
    model.fit(x_train_new, y_train_new, epochs=1, batch_size=128, verbose=1)
    loss, acc = model.evaluate(x_pool_new, y_pool_new)

    assert acc > 0.45