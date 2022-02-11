#!/usr/bin/env python -W ignore::DeprecationWarning

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import time
from multiprocessing import Pool, freeze_support
from pathlib import Path
import pathlib, pickle
from functools import partial

from sklearn.metrics import recall_score, f1_score


import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from discrepancies import datasets, pool, pool2graph, clamp_evaluation, discrepancies_intervals

import configparser

config = configparser.ConfigParser()
config.read('config_clamps.ini')


RANDOM_STATE = 42


def mc_sampling(n_sampling, dim):
    _alphas = np.random.random(n_sampling)
    _alphas = np.tile(_alphas.reshape(n_sampling, -1), (1, dim))
    return _alphas

def setup_experiment(_DATASETS, _POOL, _K_INIT, _K_REFINEMENT, _MAX_EPOCHS, _N_SAMPLING, _MAX_DELTA_ACCURACIES, OUTPUT_DIR, time_left_for_this_task=30, n_jobs=-1, RANDOM_STATE=None):

    _dataset, _pool, _k_init, _k_refinement, _max_epochs, _n_sampling, _run_suffix = [], [], [], [], [], [], []

    DATASETS, MC_SAMPLING, POOLS = {}, {}, {}

    for d in _DATASETS:

        # Get dataset
        DATASETS[d] = {}
        DATASETS[d]['X_train'], DATASETS[d]['X_test'], DATASETS[d]['y_train'], DATASETS[d]['y_test'], DATASETS[d]['scaler'], DATASETS[d]['feature_names'], DATASETS[d]['target_names'], DATASETS[d]['categorical_names'] = datasets.get_dataset(dataset=d, n_samples=1000, noise=0.3, RANDOM_STATE=RANDOM_STATE)
                
        # Draw MC eval samples
        MC_SAMPLING[d]={}
        MC_SAMPLING[d]['samples'] = mc_sampling(_N_SAMPLING, DATASETS[d]['X_train'].shape[1])

        POOLS[d] = {}
        
        for p in _POOL:

            # Init+fit pool
            if p == 'Basic':
                pool_run = pool.BasicPool(RANDOM_STATE=RANDOM_STATE)
                pool_run = pool_run.fit(DATASETS[d]['X_train'], DATASETS[d]['y_train'])
                pool_run = pool_run.filter_accuracies(DATASETS[d]['X_test'], DATASETS[d]['y_test'], _MAX_DELTA_ACCURACIES)
                print(pool_run.get_performances(DATASETS[d]['X_test'], DATASETS[d]['y_test']))
            elif p == 'AutoGluon':
                pool_run = pool.AutogluonPool(max_delta_accuracies=_MAX_DELTA_ACCURACIES)
                pool_run = pool_run.fit(DATASETS[d]['X_train'], DATASETS[d]['y_train'], output_directory=None)
                print(pool_run.get_performances(DATASETS[d]['X_train'], DATASETS[d]['y_train'])[1])
            elif p == 'AutoSklearn':
                pool_run = pool.AutoSklearnPool(max_delta_accuracies=_MAX_DELTA_ACCURACIES, time_left_for_this_task=time_left_for_this_task, n_jobs=n_jobs)
                pool_run = pool_run.fit(DATASETS[d]['X_train'], DATASETS[d]['y_train'])
            else:
                raise ValueError
                
            POOLS[d][p] = pool_run

            
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


def test_fidelity(p2g, run, pool, output_dir):
    """
    Assess the discovery of discrepancies
    """
    OUTPUT_DIR = output_dir
   
    cf_intervals = discrepancies_intervals.get_discrepancies_intervals(p2g.G)
    
    _alphas = np.random.random(run.n_sampling)
    _alphas = np.tile(_alphas.reshape(run.n_sampling, -1), (1, cf_intervals[0].border_features.shape[1]))
    
    precision_all, precision_without_disctrain = clamp_evaluation.evaluate_intervals(cf_intervals, pool, run.n_sampling, _alphas)
    n_generated = len([n for n in p2g.G.nodes if n < 0])
    
    df = pd.DataFrame({"precision_all":[precision_all], "precision_without_disctrain":[precision_without_disctrain], "n_generated":[n_generated]})
    df.to_feather(OUTPUT_DIR+'/'+str(run.run_suffix)+'_PREDS.feather')

    
        
def exec_run(run, datasets, pools, output_dir):
    DATASETS = datasets
    POOLS = pools
    i_run = run[0]
    run = run[1]
    print('### '+str(i_run)+' -- '+str(run.run_suffix))
    
    #DATASETS = dMan.copy()
    p2g = pool2graph.pool2graph(DATASETS[run.dataset]['X_train'],
                                DATASETS[run.dataset]['y_train'],
                                POOLS[run.dataset][run.pool],
                                k_init=run.k_init)#,
                                #k_refinement=run.k_refinement)
    
    p2g.fit(max_epochs=run.max_epochs)
    

    # with open(OUTPUT_DIR+'p2g$'+run.run_suffix+'.pickle', 'wb') as f:
    #     pickle.dump(p2g, f, pickle.HIGHEST_PROTOCOL)

    test_fidelity(p2g, run, POOLS[run.dataset][run.pool], output_dir)

    
    
def main():
    for expe in config.sections():
    
        N_JOBS = int(config[expe]['N_JOBS'])
        N_REPLICATION = int(config[expe]['N_REPLICATION'])
        N_SAMPLING = int(config[expe]['N_SAMPLING'])
        POOL = config[expe]['POOL'].split(',')
        DATASETS = config[expe]['DATASETS'].split(',')
        
        K_INIT = [int(i) for i in config[expe]['K_INIT'].split(',')]
        K_REFINEMENT = [int(i) for i in config[expe]['K_REFINEMENT'].split(',')]
        MAX_EPOCHS = [int(i) for i in config[expe]['MAX_EPOCHS'].split(',')]
        MAX_DELTA_ACCURACIES = float(config[expe]['MAX_DELTA_ACCURACIES']) #?

        time_expe = int(time.time())
        for i in range(N_REPLICATION):
            print('============== ITERATION %i'%i)
            
            OUTPUT_DIR = pathlib.Path(str(pathlib.Path('../..').resolve())+'/results_clamps/'+str(time_expe)+'#'+str(expe)+'_'+str(i))
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR = str(OUTPUT_DIR)+'/'

            df_expe_plan, DATASETS, MC_SAMPLING, POOLS = setup_experiment(DATASETS, POOL, K_INIT, K_REFINEMENT, MAX_EPOCHS, N_SAMPLING, MAX_DELTA_ACCURACIES, OUTPUT_DIR, time_left_for_this_task=30, n_jobs=-1, RANDOM_STATE=i)
            runs = df_expe_plan.iterrows()

            with Pool(N_JOBS) as p: ### Conséquence de ça: il vaut mieux mettre un nombre d'expes par itération ( K * epochs * dataset * etc. ) qui soit un multiple du nombre de workers max. Car sinon il laisse des workers rien faire en attendant d'avoir fini la première itération.
                p.map(partial(exec_run, datasets=DATASETS, pools=POOLS, output_dir=OUTPUT_DIR), runs)    
    

if __name__ == "__main__":
    """ Start experiment with multiprocessing
    """
    freeze_support()
    main()
 

    