import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import time
from multiprocessing import Pool
from pathlib import Path
import pathlib, pickle

from sklearn.metrics import f1_score
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

import configparser

config = configparser.ConfigParser()
config.read('config.ini')


RANDOM_STATE = 42


def mc_sampling(n_sampling, X_train):

    X_samples, kde_score = evaluation.random_sampling_kde(X_train, n=n_sampling)
    X_samples = pd.DataFrame(X_samples, columns=X_train.columns)
    
    return X_samples


def setup_experiment(_DATASETS, _POOL, _K_INIT, _K_REFINEMENT, _MAX_EPOCHS, _N_SAMPLING, _MAX_DELTA_ACCURACIES, time_left_for_this_task=30, n_jobs=-1):

    _dataset, _pool, _k_init, _k_refinement, _max_epochs, _n_sampling, _run_suffix = [], [], [], [], [], [], []

    DATASETS, MC_SAMPLING, POOLS = {}, {}, {}

    for d in _DATASETS:

        # Get dataset
        DATASETS[d] = {}
        DATASETS[d]['X_train'], DATASETS[d]['X_test'], DATASETS[d]['y_train'], DATASETS[d]['y_test'], DATASETS[d]['scaler'], DATASETS[d]['feature_names'], DATASETS[d]['target_names'] = datasets.get_dataset(dataset=d, n_samples=1000, noise=0.3)
                
        # Draw MC eval samples
        MC_SAMPLING[d]={}
        MC_SAMPLING[d]['samples'] = mc_sampling(_N_SAMPLING, DATASETS[d]['X_train'])

        POOLS[d] = {}
        
        for p in _POOL:

            # Init+fit pool
            if p == 'Basic':
                pool_run = pool.BasicPool()
                pool_run = pool_run.fit(DATASETS[d]['X_train'], DATASETS[d]['y_train'])
            elif p == 'AutoGluon':
                pool_run = pool.AutogluonPool()
                pool_run = pool_run.fit(DATASETS[d]['X_train'], DATASETS[d]['y_train'], output_directory=None)
            elif p == 'AutoSklearn':
                pool_run = pool.AutoSklearnPool(max_delta_accuracies=_MAX_DELTA_ACCURACIES, time_left_for_this_task=time_left_for_this_task, n_jobs=n_jobs)
                pool_run = pool_run.fit(DATASETS[d]['X_train'], DATASETS[d]['y_train'])
            else:
                raise ValueError
                
            POOLS[d][p] = pool_run
            
            #Label MC eval samples
            MC_SAMPLING[d]['labels'] = pool_run.predict_discrepancies(MC_SAMPLING[d]['samples'])
            
            for ki in _K_INIT:
                for kr in _K_REFINEMENT:
                    for me in _MAX_EPOCHS:

                        _dataset.append(d)
                        _pool.append(p)
                        _k_init.append(ki)
                        _k_refinement.append(kr)
                        _max_epochs.append(me)

                        _n_sampling.append(_N_SAMPLING)
                        _run_suffix.append('D$'+str(d)+'_P$'+str(p)+'_KI$'+str(ki)+'_KR$'+str(kr)+'_ME$'+str(me))

                        
    # with open(OUTPUT_DIR+'DATASETS.pickle', 'wb') as f:
    #     pickle.dump(DATASETS, f, pickle.HIGHEST_PROTOCOL)
    
    # with open(OUTPUT_DIR+'MC_SAMPLING.pickle', 'wb') as f:
    #     pickle.dump(MC_SAMPLING, f, pickle.HIGHEST_PROTOCOL)
        
    # with open(OUTPUT_DIR+'POOLS.pickle', 'wb') as f:
    #     pickle.dump(POOLS, f, pickle.HIGHEST_PROTOCOL)
                        
    df_expe_plan = {
        'dataset':_dataset,
        'pool':_pool,
        'k_init':_k_init,
        'k_refinement':_k_refinement,
        'max_epochs':_max_epochs,
        'n_sampling':_n_sampling,
        'run_suffix':_run_suffix,
    }
    
    df_expe_plan = pd.DataFrame(df_expe_plan)
    df_expe_plan.to_feather(OUTPUT_DIR+'expe_plan.feather')

    return df_expe_plan, DATASETS, MC_SAMPLING, POOLS


def test_fidelity(p2g, run):
    """
    Assess the discovery of discrepancies
    """

    # Get the dataset of discrepancies from the pool2graph
    X_discr, y_discr = p2g.get_discrepancies_dataset()

    # Instantiate models that will be trained on the dataset of discrepancies
    models = {}
    models['P2G-xgb'] = xgb.XGBClassifier(n_jobs=1, verbosity=0).fit(X_discr, y_discr)
    models['P2G-tree'] = DecisionTreeClassifier(random_state=RANDOM_STATE, max_leaf_nodes=10).fit(X_discr, y_discr)
    models['P2G-rfc'] = RandomForestClassifier(n_jobs=1).fit(X_discr, y_discr)
    
    ## Evaluate "fidelity" i.e. capacity of the graph to discover discrepancies. This is done by (1) training a black box on the dataset of discrepancies from the trained pool2graph. The objective of the blackbox is to predict if an instance is in an area of discrepancies or not. (2) The assessment of the pool2graph fidelity to discover the areas of discrepancies of the pool is done by Monte Carlo simulation. A random sampling of _N_SAMPLING points from the original distribution is done, points are labeled by the pool (discrepancy or not) this new dataset is the ground truth. The blackbox trained on the discrepancies' dataset from the pool2graph is used to predicted discrepancies for the points resulting from the random sampling. Fidelity is assessed with the ground truth label of the sampled points.
    ## Important: One single sample of points is drawn by dataset, so are the pools: trained once by dataset (and thus shared across experiments - set of parameters). Results of different experiments are thus comparable on the same dataset.
    
    # Prepare dict that will receive results
    
    X_sampling = MC_SAMPLING[run.dataset]['samples']
    y_sampling = MC_SAMPLING[run.dataset]['labels']
    
    preds_sampling = {}
    preds_sampling['y_pool_discr'] = y_sampling

    # For sampled points, blackboxes trained on the discrepancies' dataset predict if they are in discrepancies areas or not
    for k in models.keys():
        preds_sampling[k] = models[k].predict(X_sampling)

    # Test if a simple random sampling (following X_train distrib) labeled by the pool (for presence/absence of discrepancies) is as performant
    # /!\ should be done several times to counter-effect randomness?
    X_samples, kde_score = evaluation.random_sampling_kde(DATASETS[run.dataset]['X_train'], n=len(X_discr))
    X_samples = pd.DataFrame(X_samples, columns=DATASETS[run.dataset]['X_train'].columns)
    y_samples_pool_discr = p2g.pool.predict_discrepancies(X_samples)
    
    models = {}
    models['RandomSampling-xgb'] = xgb.XGBClassifier(n_jobs=1, verbosity=0).fit(X_samples, y_samples_pool_discr)
    models['RandomSampling-tree'] = DecisionTreeClassifier(random_state=RANDOM_STATE, max_leaf_nodes=10).fit(X_samples, y_samples_pool_discr)
    models['RandomSampling-rfc'] = RandomForestClassifier(n_jobs=1).fit(X_samples, y_samples_pool_discr)
    
    for k in models.keys():
        preds_sampling[k] = models[k].predict(X_sampling)

    df = pd.DataFrame(preds_sampling)
    df.to_feather(OUTPUT_DIR+'/'+str(run.run_suffix)+'_PREDS.feather')


def exec_run(run):
    i_run = run[0]
    run = run[1]
    print('### '+str(i_run)+' -- '+str(run.run_suffix))
    
    p2g = pool2graph.pool2graph(DATASETS[run.dataset]['X_train'],
                                DATASETS[run.dataset]['y_train'],
                                POOLS[run.dataset][run.pool],
                                k_init=run.k_init,
                                k_refinement=run.k_refinement)
    
    p2g.fit(max_epochs=run.max_epochs)
    

    # with open(OUTPUT_DIR+'p2g$'+run.run_suffix+'.pickle', 'wb') as f:
    #     pickle.dump(p2g, f, pickle.HIGHEST_PROTOCOL)

    test_fidelity(p2g, run)


if __name__ == "__main__":
    """ Start experiment with multiprocessing
    """

    for expe in config.sections():
    
        N_JOBS = int(config[expe]['N_JOBS'])
        N_REPLICATION = int(config[expe]['N_REPLICATION'])
        N_SAMPLING = int(config[expe]['N_SAMPLING'])
        POOL = config[expe]['POOL'].split(',')
        DATASETS = config[expe]['DATASETS'].split(',')
        
        K_INIT = [int(i) for i in config[expe]['K_INIT'].split(',')]
        K_REFINEMENT = [int(i) for i in config[expe]['K_REFINEMENT'].split(',')]
        MAX_EPOCHS = [int(i) for i in config[expe]['MAX_EPOCHS'].split(',')]
        MAX_DELTA_ACCURACIES = float(config[expe]['MAX_DELTA_ACCURACIES'])

        time_expe = int(time.time())
        for i in range(N_REPLICATION):
            
            OUTPUT_DIR = pathlib.Path(str(pathlib.Path('../..').resolve())+'/results/'+str(time_expe)+'#'+str(expe)+'_'+str(i))
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR = str(OUTPUT_DIR)+'/'

            df_expe_plan, DATASETS, MC_SAMPLING, POOLS = setup_experiment(DATASETS, POOL, K_INIT, K_REFINEMENT, MAX_EPOCHS, N_SAMPLING, MAX_DELTA_ACCURACIES, time_left_for_this_task=30, n_jobs=-1)
            runs = df_expe_plan.iterrows()

            with Pool(N_JOBS) as p:
                p.map(exec_run, runs)

    