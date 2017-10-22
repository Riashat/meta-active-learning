# combining_bayesian_al_ssl
Combining Bayesian Active Learning with Semi-Supervised Learning


## Requirements
Python 3.5.2

Keras version : 2.0.8

Theano : '0.9.0'

scikit-learn : '0.19.0'

## To run `mnist_bayesian_al_ssl_clustering.py`

```
python3 mnist_bayesian_al_ssl_clustering.py -g GPU -o ORACLE_TYPE -e EPOCHS -d DROPOUT_ITERATIONS
```

ORACLE_TYPE could be: `nearestpoint` or `nearestlabel`

- `nearestpoint` will return the point and label that are closest to the queried points
- `nearestlabel` will return the label for the point that is closest to the queried point

