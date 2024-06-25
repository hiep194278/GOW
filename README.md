# Generalized Ordered Wasserstein

## Dependencies

GOW relies on the following dependencies:

- `numpy`
- `scikit-learn`
- `POT`
- `aeon`
- `joblib`
- `seaborn`
- `matplotlib`

## Installation

Install GOW by to cloning it using the Web URL:

```
$ git clone https://github.com/hiep194278/GOW.git
```

Alternatively, you can download as ZIP and then unzip it to your folder.

## Usage

The repository provides functions to measure the similarity between two temporal sequences using monotonic functions.

```python
from gow.utilities import load_ucr_dataset
from gow import gow_sinkhorn_autoscale
from ot import dist

# Get UCR train and test sets
X_train, y_train, X_test, y_test = load_ucr_dataset("../data/UCR", "Chinatown")

seq1 = X_test[2]
seq2 = X_train[3]

# Compute the initial cost matrix
D = dist(seq1, seq2)
D = D / D.max()

# Distance returned by GOW
gow_sinkhorn_autoscale([], [], D)
```

You can also run the k-NN classifier with the distance matrix computed by GOW.

```python
from gow.utilities import load_human_action_dataset

# Get Human Actions train and test sets
X_train, y_train, X_test, y_test = load_human_action_dataset("../data/Human_Actions", "Weizmann")

# Run k-NN classifier with default k values of 1, 3 and 5
run_knn(X_train, y_train, X_test, y_test)
```

## Examples

You can find some simple examples on our [examples
page](https://github.com/hiep194278/GOW/tree/main/examples) and an
example [jupyter
notebook](https://github.com/hiep194278/GOW/blob/main/examples/example.ipynb).

## Help

If you have any questions about GOW or encounter difficulties when trying
to build the tool on your own system, please open an issue in the project's
[issue tracker](https://github.com/hiep194278/GOW/issues). Provide a detailed
description of the issue so that we can better assist you.

## Contributors

GOW is developed and maintained by:

- Hiep To ([GitHub](https://github.com/hiep194278))
- Tung Doan ([GitHub](https://github.com/TungDP))

<!-- ## Citation
```
``` -->
