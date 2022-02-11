import numpy as np
import pandas as pd

import sys, os
# add folder path where discrepancies folder is
#_PATH = '/Users/a435vv/OneDrive - AXA/Projects/BlackboxesDiscrepancies/discrepancies-in-machine-learning/'
#sys.path.insert(0, _PATH) 
sys.path.append(os.path.dirname(sys.path[0]))

from discrepancies import pool





def evaluate_intervals(intervals, pool, n, alphas):
    # score for n intervals
    precision_all = []
    precision_without_disctrain = []
    
    for interval in intervals:
            
        first_d, last_d = interval.border_discrepancies[0], interval.border_discrepancies[-1]
        
        if (first_d == False) or (last_d == False):
            precision_without_disctrain.append(clamps_score(interval, pool, alphas, n))
        
        precision_all.append(clamps_score(interval, pool, alphas, n))
        
        
    return precision_all, precision_without_disctrain


def clamps_score(interval, pool, _alphas, n):
    # score for one interval
    segment = interval.border_features

    a0, a1 = segment.iloc[0], segment.iloc[1]
    new_obs = np.multiply(np.tile(a0, (n, 1)), _alphas) + np.multiply(np.tile(a1, (n, 1)), 1 - _alphas)
    _preds = pool.predict_discrepancies(np.array(new_obs))
    return _preds.mean()