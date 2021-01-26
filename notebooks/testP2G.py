import numpy as np
import pandas as pd

import time
from multiprocessing import Pool

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import sys, os
# add folder path where discrepancies folder is
#_PATH = '/Users/a435vv/OneDrive - AXA/Projects/BlackboxesDiscrepancies/discrepancies-in-machine-learning/'
#sys.path.insert(0, _PATH) 
sys.path.append(os.path.dirname(sys.path[0]))

from discrepancies import datasets, pool, pool2graph, evaluation

RANDOM_STATE = 42

"""
PARAMETERS OF THE EXPERIMENT
"""

N_JOBS = 15
_OUTPUT_DIRECTORY = '/home/ec2-user/SageMaker/results'

_N_REPLICATION = 3

#_POOL = ['Basic']
_POOL = ['AutoGluon']
#_POOL = ['Basic', 'AutoGluon']

# _DATASETS = ['half-moons', 'breast-cancer', 'load-wine', 'kddcup99']
#_DATASETS = ['half-moons']
_DATASETS = ['half-moons', 'breast-cancer', 'load-wine']

_K_INIT = [1,3,5,10]
_K_REFINEMENT = [0,1,3,5,10]
_MAX_EPOCHS = [0,1,2,3,4,5]
stopping_criterion = 0.01


def create_expe(_POOL, _DATASETS, _K_INIT, _K_REFINEMENT, _MAX_EPOCHS, _N_REPLICATION):

    _P2G_SETUPS = {}
    
    for p in _POOL:
        for d in _DATASETS:
            for k_init in _K_INIT:
                for k_refinement in _K_REFINEMENT:
                    for max_epochs in _MAX_EPOCHS:
                        for n_replication in range(_N_REPLICATION):

                            # run_name = 'p'+str(p)+'_d'+str(d)+'_ki'+str(k_init)+'_kr'+str(k_refinement)+'_e'+str(max_epochs)+'_s'+str(stopping_criterion)+'_r'+str(time.time())
                            run_name = 'p$'+str(p)+'_d$'+str(d)+'_ki$'+str(k_init)+'_kr$'+str(k_refinement)+'_e$'+str(max_epochs)+'_s$'+str(stopping_criterion)+'_r$'+str(n_replication)
                            _P2G_SETUPS[run_name] = {'pool':p,
                                                'dataset':d,
                                                'k_init':k_init,
                                                'k_refinement':k_refinement,
                                                'max_epochs':max_epochs,
                                                'stopping_criterion':stopping_criterion}
                        
    return _P2G_SETUPS


"""
DEFINE ONE RUN
"""

def run(cfg_i):

    print('#### Start Run #'+str(cfg_i))

    cfg = _P2G_SETUPS[list(_P2G_SETUPS.keys())[cfg_i]]
    run_name = list(_P2G_SETUPS.keys())[cfg_i]

    print(cfg)

    X_train, X_test, y_train, y_test, scaler, feature_names, target_names = datasets.get_dataset(dataset=cfg['dataset'], n_samples=1000, noise=0.3)

    pool_name = cfg['pool']

    if pool_name == 'Basic':
        pool_run = pool.BasicPool()
        pool_run = pool_run.fit(X_train, y_train)
    elif pool_name == 'AutoGluon':
        pool_run = pool.AutogluonPool()
        pool_run = pool_run.fit(X_train, y_train, output_directory=_OUTPUT_DIRECTORY+'Autogluon/')
    else:
        raise ValueError

    p2g = pool2graph.pool2graph(X_train, y_train, pool_run, k_init=cfg['k_init'], k_refinement=cfg['k_refinement'])
    p2g.fit(max_epochs=cfg['max_epochs'], stopping_criterion=cfg['stopping_criterion'])

    # cfg['fidelity'] = 

    s = pd.Series(cfg)
    s.to_csv(_OUTPUT_DIRECTORY+'/'+str(run_name)+'.csv')

    print('---- End Run #'+str(cfg_i))

    return cfg


"""
RUN EXPE
"""


if __name__ == "__main__":

    _POOL_TMP = [p for p in _POOL if p != 'AutoGluon']
    if len(_POOL_TMP)>0:
        _P2G_SETUPS = create_expe(_POOL_TMP, _DATASETS, _K_INIT, _K_REFINEMENT, _MAX_EPOCHS, _N_REPLICATION)
        runs = range(len(list(_P2G_SETUPS.keys())))

        with Pool(N_JOBS) as p:
            p.map(run, runs)

        
    # For Autogluon models: autogluon is already doing parallelization => no Pool at the experiments level when autogluon models
    if 'AutoGluon' in _POOL:
        _POOL_TMP = ['AutoGluon']
        _P2G_SETUPS = create_expe(_POOL_TMP, _DATASETS, _K_INIT, _K_REFINEMENT, _MAX_EPOCHS, _N_REPLICATION)
        runs = range(len(list(_P2G_SETUPS.keys())))
    
        for r in runs:
            cfg = run(r)