from src import datatools
from src.networks import cnn
from src.oracle import ask_oracle
from src.acquisition_function import maxentropy

def test_aquisition():

    (x_train, y_train), val_data, (x_pool, y_pool), test_data = datatools.data_pipeline()
    n_classes = y_train.shape[1]

    model = cnn(input_shape=x_train.shape[1:],
                output_classes=n_classes)



    pool_without_subset, (x_pool_subset, y_pool_subset) = datatools.get_pool_subset(x_pool, y_pool, 2000)

    assert pool_without_subset[0].shape[0] == x_pool.shape[0]-2000
    assert x_pool_subset.shape[0] == 2000

    uncertainty_estimates = maxentropy(x_pool_subset, n_classes, model, dropout_iterations=2)

    new_data_for_training, pool_subset_updated = ask_oracle(uncertainty_estimates, 5, x_pool_subset, y_pool_subset, n_classes=n_classes)
    x_train_new, y_train_new = datatools.combine_datasets(new_data_for_training, (x_train, y_train))
    x_pool_new, y_pool_new = datatools.combine_datasets(pool_without_subset, pool_subset_updated)

    assert x_train_new.shape[0] == x_train.shape[0] + 5, 'Training data increases by 5 points only'
    assert x_pool_new.shape[0] == x_pool.shape[0] - 5, 'Pool must reduce by 5 points only'