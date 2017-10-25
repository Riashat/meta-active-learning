# combining_bayesian_al_ssl
Combining Bayesian Active Learning with Semi-Supervised Learning


## Requirements
Python 3.5.2

See `requirements.txt`

You can install as `pip3.5 install -r requirements.txt --user`

## Tests
Some unit tests exist
run `python3 -m pytest` on linux or `pytest` on os x.

## Running

```
python3 mnist_bayesian_al_ssl_clustering.py -g GPU -o ORACLE_TYPE -e EPOCHS -d DROPOUT_ITERATIONS
```

ORACLE_TYPE could be: `nearestpoint` or `nearestlabel`

- `nearestpoint` will return the point and label that are closest to the queried points
- `nearestlabel` will return the label for the point that is closest to the queried point

