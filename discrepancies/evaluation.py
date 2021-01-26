import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, f1_score


def get_fit_evaluation(xpl, X_discr, y_discr, verbose=True):

    skf = StratifiedKFold(n_splits=10)

    accuracy = make_scorer(accuracy_score)

    balanced_accuracy = make_scorer(balanced_accuracy_score)
    f1_score_non_discrepancies = make_scorer(f1_score, average='binary', pos_label=0)
    f1_score_discrepancies = make_scorer(f1_score, average='binary', pos_label=1)

    scores = cross_validate(xpl, X_discr, y=y_discr, cv=skf, return_train_score=True, return_estimator=False, scoring={'accuracy':accuracy, 'balanced_accuracy':balanced_accuracy, 'f1_score_non_discrepancies':f1_score_non_discrepancies, 'f1_score_discrepancies':f1_score_discrepancies}, n_jobs=-1)

    estimator = xpl.fit(X_discr, y_discr)

    scores = pd.DataFrame(scores).mean().round(2)
    scores.name = str(xpl)
    
    return scores, estimator


def random_sampling_kde(X, n=1000, kernel='gaussian', bandwidth=0.2):

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kde = kde.fit(X)

    score = kde.score_samples(X)

    X_samples = kde.sample(n)

    return X_samples, score


def get_fidelity_evaluation(xpl, X_samples, y_pool_discr):

    y_xpl_discr = predict_discrepancies(X_samples, xpl)

    accuracy = accuracy_score(y_pool_discr, y_xpl_discr)
    balanced_accuracy = balanced_accuracy_score(y_pool_discr, y_xpl_discr)
    f1_score_discr = f1_score(y_pool_discr, y_xpl_discr, pos_label=1, average='binary')
    f1_score_non_discr = f1_score(y_pool_discr, y_xpl_discr, pos_label=0, average='binary')

    scores = {'accuracy_discrepancy':accuracy, 'balanced_accuracy_discrepancy':balanced_accuracy, 'f1-score_discrepancies':f1_score_discr, 'f1-score_NON_discrepancies':f1_score_non_discr}
    scores = pd.Series(scores, name=str(xpl)).round(2)

    return scores


def predict_discrepancies(X, xpl):
    preds = xpl.predict(X)
    preds_discr = preds>0
    return preds_discr.astype('int')


def plot_tree(xpl_tree):
    tree.plot_tree(xpl_tree)
    plt.show()


def compare_fit_explainer(xpls, X_discr, y_discr):

    fit_scores, estimators = {}, {}

    for xpl in xpls:
        fit_scores[str(xpl)], estimators[str(xpl)] = get_fit_evaluation(xpl, X_discr, y_discr)

    return pd.DataFrame(fit_scores), estimators


def compare_fidelity_explainer(xpls, p2g, X_train, n=1000):

    X_samples, kde_score = random_sampling_kde(X_train, n=n)
    y_pool_discr = p2g.pool.predict_discrepancies(X_samples)

    fidelity_scores = {}

    for xpl in xpls:
        fidelity_score = get_fidelity_evaluation(xpls[xpl], X_samples, y_pool_discr)

        fidelity_scores[xpl] = fidelity_score

    return pd.DataFrame(fidelity_scores)

