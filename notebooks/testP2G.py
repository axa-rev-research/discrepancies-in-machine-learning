import numpy as np
import pandas as pd

import time
from multiprocessing import Pool

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold

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
_N_SAMPLING = 5000

_POOL = ['Basic']
#_POOL = ['AutoGluon']
#_POOL = ['Basic', 'AutoGluon']

# _DATASETS = ['half-moons', 'breast-cancer', 'load-wine', 'kddcup99']
_DATASETS = ['half-moons']
#_DATASETS = ['half-moons', 'breast-cancer', 'load-wine']
#_DATASETS = ['boston', 'credit-card', 'churn']

_K_INIT = [1,3,5,10]
_K_REFINEMENT = [0,1,3,5,10]
_MAX_EPOCHS = [0,1,2,3,4,5]
stopping_criterion = 0.01


#def create_expe(_POOL, _DATASETS, _K_INIT, _K_REFINEMENT, _MAX_EPOCHS, _N_REPLICATION):

_P2G_SETUPS = {}

_X_train = {}
_X_test = {}
_y_train = {}
_y_test = {}
_scaler = {}
_feature_names = {}
_target_names = {}

_pools = {}

_X_sampling_fidelity_eval = {}
_y_sampling_fidelity_eval = {}

flag_first_iter = True
for p in _POOL:
    for d in _DATASETS:

        if flag_first_iter:
            _X_train[d] = {}
            _X_test[d] = {}
            _y_train[d] = {}
            _y_test[d] = {}
            _scaler[d] = {}
            _feature_names[d] = {}
            _target_names[d] = {}

            _pools[d] = {}
            _X_sampling_fidelity_eval[d] = {}
            _y_sampling_fidelity_eval[d] = {}

            flag_first_iter = False

        X_train, X_test, y_train, y_test, scaler, feature_names, target_names = datasets.get_dataset(dataset=d, n_samples=1000, noise=0.3)
        _X_train[d][p] = X_train
        _X_test[d][p] = X_test
        _y_train[d][p] = y_train
        _y_test[d][p] = y_test
        _scaler[d][p] = scaler
        _feature_names[d][p] = feature_names
        _target_names[d][p] = target_names

        if p == 'Basic':
            pool_run = pool.BasicPool()
            pool_run = pool_run.fit(X_train, y_train)
        elif p == 'AutoGluon':
            pool_run = pool.AutogluonPool()
            pool_run = pool_run.fit(X_train, y_train, output_directory=_OUTPUT_DIRECTORY+'Autogluon/')
        else:
            raise ValueError

        _pools[d][p] = pool_run

        X_samples, kde_score = evaluation.random_sampling_kde(X_train, n=_N_SAMPLING)
        X_samples = pd.DataFrame(X_samples, columns=X_train.columns)
        y_pool_discr = pool_run.predict_discrepancies(X_samples)

        _X_sampling_fidelity_eval[d][p] = X_samples
        _y_sampling_fidelity_eval[d][p] = y_pool_discr

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
                    
#    return _P2G_SETUPS


def test_fidelity(p2g, run_name, cfg):

    X_discr, y_discr = p2g.get_discrepancies_dataset()

    models = {}
    models['XGB'] = xgb.XGBClassifier(n_jobs=1).fit(X_discr, y_discr)
    models['tree'] = DecisionTreeClassifier(random_state=RANDOM_STATE, max_leaf_nodes=10).fit(X_discr, y_discr)
    models['RFC'] = RandomForestClassifier().fit(X_discr, y_discr)
    
    preds_test = {}
    preds_test['y_test'] = y_discr

    preds_sampling = {}
    preds_sampling['y_pool_discr'] = _y_sampling_fidelity_eval[cfg['dataset']][cfg['pool']]

    X_sampling = _X_sampling_fidelity_eval[cfg['dataset']][cfg['pool']]

    for k in models.keys():
        preds_test[k] = models[k].predict(X_discr)
        preds_sampling[k] = models[k].predict(X_sampling)

    # Test a dumb random sampling (following X_train distrib)
    X_samples, kde_score = evaluation.random_sampling_kde(X_train, n=len(X_discr))
    X_samples = pd.DataFrame(X_samples, columns=X_train.columns)
    y_samples_pool_discr = pool_run.predict_discrepancies(X_samples)
    XGB = xgb.XGBClassifier(n_jobs=1).fit(X_samples, y_samples_pool_discr)
    preds_sampling['Random Sampling'] = XGB.predict(X_sampling)

    df1 = pd.DataFrame(preds_test)
    df1.to_csv(_OUTPUT_DIRECTORY+'/'+str(run_name)+'_PREDS_train.csv')

    df2 = pd.DataFrame(preds_sampling)
    df2.to_csv(_OUTPUT_DIRECTORY+'/'+str(run_name)+'_PREDS_sampling.csv')

    return len(X_discr)


"""
DEFINE ONE RUN
"""

def run(cfg_i):

    print('#### Start Run #'+str(cfg_i))

    cfg = _P2G_SETUPS[list(_P2G_SETUPS.keys())[cfg_i]]
    run_name = list(_P2G_SETUPS.keys())[cfg_i]

    print(cfg)

    dataset = cfg['dataset']
    pool_name = cfg['pool']

    X_train =_X_train[dataset][pool_name]
    X_test = _X_test[dataset][pool_name]
    y_train = _y_train[dataset][pool_name]
    y_test = _y_test[dataset][pool_name]
    scaler = _scaler[dataset][pool_name]
    feature_names = _feature_names[dataset][pool_name]
    target_names = _target_names[dataset][pool_name]

    p2g = pool2graph.pool2graph(X_train, y_train, pool_run, k_init=cfg['k_init'], k_refinement=cfg['k_refinement'])
    p2g.fit(max_epochs=cfg['max_epochs'], stopping_criterion=cfg['stopping_criterion'])

    s = pd.Series(cfg)
    s.to_csv(_OUTPUT_DIRECTORY+'/'+str(run_name)+'_CONFIG.csv')

    # cfg['fidelity'] = 

    n_X_discr = test_fidelity(p2g, run_name, X_train, cfg)

    print('---- End Run #'+str(cfg_i))

    return cfg


"""
RUN EXPE
"""


if __name__ == "__main__":


    runs = range(len(list(_P2G_SETUPS.keys())))

    with Pool(N_JOBS) as p:
        p.map(run, runs)

    