import os
from os.path import basename, join
import sys
import joblib

import numpy as np
import ot
from aeon.datasets import load_from_tsfile
from sklearn import neighbors
from sklearn.metrics import accuracy_score

sys.path.append('src')
from gow import gow_sinkhorn_autoscale

def load_human_action_dataset(data_dir, dataset_name):
    X_train = joblib.load(os.path.join(data_dir, dataset_name, "X_train.pkl"))
    y_train = joblib.load(os.path.join(data_dir, dataset_name, "y_train.pkl"))
    X_test = joblib.load(os.path.join(data_dir, dataset_name, "X_test.pkl"))
    y_test = joblib.load(os.path.join(data_dir, dataset_name, "y_test.pkl"))

    print("Successfully loaded dataset:", dataset_name)
    print("Size of train data:", len(y_train))
    print("Size of test data:", len(y_test))

    return X_train, y_train, X_test, y_test

def fix_array(X):
    X_new = np.empty((X.shape[0], X.shape[2], X.shape[1]))

    for i in range (X_new.shape[0]):
        X_new[i] = np.transpose(X[i])

    return X_new

def load_ucr_dataset(data_dir, dataset_name):
    X, y_train = load_from_tsfile(os.path.join(data_dir, dataset_name, f'{dataset_name}_TRAIN'))
    X_train = fix_array(X)
    X, y_test = load_from_tsfile(os.path.join(data_dir, dataset_name, f'{dataset_name}_TEST'))
    X_test = fix_array(X)

    print("Successfully loaded dataset:", dataset_name)
    print("Size of train data:", len(y_train))
    print("Size of test data:", len(y_test))

    return X_train, y_train, X_test, y_test

def run_knn(X_train, y_train, X_test, y_test, normalize_cost_matrix=True, cost_metric="minkowski", num_neighbor_list=[1, 3, 5]):
    train_len = len(y_train)
    test_len = len(y_test)
    X_computed = np.ones((train_len, train_len))
    X_test_computed = np.empty((test_len, train_len))

    for i in range(test_len):
        print("Batch:", str(i+1) + "/" + str(test_len))
        for j in range(train_len):
            C = ot.dist(X_test[i], X_train[j], metric=cost_metric)
            if normalize_cost_matrix:
                C = C / C.max()

            X_test_computed[i][j] = gow_sinkhorn_autoscale([], [], C)


    for k in num_neighbor_list:
        clf = neighbors.KNeighborsClassifier(n_neighbors = k, metric="precomputed")
        clf.fit(X_computed, y_train)
        y_pred = clf.predict(X_test_computed)
    
        print("Accuracy of " + str(k) + "NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
