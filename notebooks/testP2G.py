import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import networkx as nx

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import sys; sys.path.insert(0, '/Users/a435vv/OneDrive - AXA/Projects/BlackboxesDiscrepancies/discrepancies-in-machine-learning/') # add parent folder path where discrepancies folder is

from discrepancies import datasets, pool, pool2graph, evaluation

RANDOM_STATE = 42



# Get data and fit a pool of classifiers on it

X_train, X_test, y_train, y_test, scaler, feature_names, target_names = datasets.get_dataset(n_samples=1000, noise=0.3)
#X_train, X_test, y_train, y_test, scaler, feature_names, target_names = datasets.get_dataset(dataset='breast-cancer', n_samples=1000, noise=0.3)

pool1 = pool.BasicPool()
pool1 = pool1.fit(X_train, y_train)



p2g = pool2graph.pool2graph(X_train, y_train, pool1, k=10)
p2g.fit(max_epochs=2)

# Get discrepancies dataset (i.e. where the pool produce discrepancies according to the pool2graph)
X_discr, y_discr = p2g.get_discrepancies_dataset()

# Instantiate explainers - put them in a list
xpl_tree = DecisionTreeClassifier(random_state=RANDOM_STATE, max_leaf_nodes=20)
xpl_rfc = RandomForestClassifier()

xpls = [xpl_tree, xpl_rfc]

# Compare estimators' goodness of fit
fit_scores, xpl_estimators = evaluation.compare_fit_explainer(xpls, X_discr, y_discr)

# Compare estimators' fidelity to pool's discrepancies (drawing new samples)
fidelity_scores = evaluation.compare_fidelity_explainer(xpl_estimators, p2g, X_train, n=5000)