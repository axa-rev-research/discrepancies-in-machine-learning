import numpy as np
import pandas as pd

import sys, os
# add folder path where discrepancies folder is
#_PATH = '/Users/a435vv/OneDrive - AXA/Projects/BlackboxesDiscrepancies/discrepancies-in-machine-learning/'
#sys.path.insert(0, _PATH) 
sys.path.append(os.path.dirname(sys.path[0]))

from discrepancies import pool

from sklearn.metrics import f1_score


def discrepancy_score(p2g, run, X, pool, method='knn'):
    
    discr_true = pool.predict_discrepancies(X)
    
    discr_pred = p2g.predict_discrepancies_from_graph(X, k_neighbors=run.k_neighbors, method=method)
    return f1_score(discr_true, discr_pred)

  