import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import networkx as nx

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import sys; sys.path.insert(0, '/Users/a435vv/OneDrive - AXA/Projects/BlackboxesDiscrepancies/discrepancies-in-machine-learning/') # add parent folder path where discrepancies folder is

from discrepancies import datasets, pool, pool2graph, evaluation

RANDOM_STATE = 42


print("##### HALF MOONS #####")

X_train, X_test, y_train, y_test, scaler, feature_names, target_names = datasets.get_dataset(n_samples=1000, noise=0.3)

print("--- Basic Pool ---")
pool1 = pool.BasicPool()
pool1 = pool1.fit(X_train, y_train)

print("pool2graph.pool2graph(X_train, y_train, pool1, k=0, k_refinement_edges=0)")
p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=0)
p2g.fit(max_epochs=0, stopping_criterion=0.01)

print("pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=0)")
p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=0)
p2g.fit(max_epochs=10, stopping_criterion=0.01)

print("pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=10)")
p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=10)
p2g.fit(max_epochs=10, stopping_criterion=0.01)
"""
print("--- Autogluon Pool ---")

pool1 = pool.AutogluonPool()
pool1 = pool1.fit(X_train, y_train)

print("pool2graph.pool2graph(X_train, y_train, pool1, k=0, k_refinement_edges=0)")
p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=0, k_refinement_edges=0)
p2g.fit(max_epochs=0, stopping_criterion=0.01)

print("pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=0)")
p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=0)
p2g.fit(max_epochs=10, stopping_criterion=0.01)

print("pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=10)")
p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=10)
p2g.fit(max_epochs=10, stopping_criterion=0.01)
"""
########

for d in ['breast-cancer', 'load-wine', '20-newsgroups', 'kddcup99']:

    print("##### "+d+" #####")

    X_train, X_test, y_train, y_test, scaler, feature_names, target_names = datasets.get_dataset(dataset=d, n_samples=1000, noise=0.3)

    print("--- Basic Pool ---")
    pool1 = pool.BasicPool()
    pool1 = pool1.fit(X_train, y_train)

    print("pool2graph.pool2graph(X_train, y_train, pool1, k=0, k_refinement_edges=0)")
    p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=0, k_refinement_edges=0)
    p2g.fit(max_epochs=0, stopping_criterion=0.01)

    print("pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=0)")
    p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=0)
    p2g.fit(max_epochs=10, stopping_criterion=0.01)

    print("pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=10)")
    p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=10)
    p2g.fit(max_epochs=10, stopping_criterion=0.01)
    """
    print("--- Autogluon Pool ---")

    pool1 = pool.AutogluonPool()
    pool1 = pool1.fit(X_train, y_train)

    print("pool2graph.pool2graph(X_train, y_train, pool1, k=0, k_refinement_edges=0)")
    p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=0, k_refinement_edges=0)
    p2g.fit(max_epochs=0, stopping_criterion=0.01)

    print("pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=0)")
    p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=0)
    p2g.fit(max_epochs=10, stopping_criterion=0.01)

    print("pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=10)")
    p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=10, k_refinement_edges=10)
    p2g.fit(max_epochs=10, stopping_criterion=0.01)
    """