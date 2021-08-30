import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from pathlib import Path
import pathlib, pickle

from sklearn.metrics import f1_score


from sklearn.model_selection import StratifiedKFold

import sys, os
# sys.path.append(os.path.dirname(sys.path[0]))

# from discrepancies import datasets, pool, pool2graph, evaluation


def get_all_expe():
    
    OUTPUT_DIR = pathlib.Path(str(pathlib.Path('../..').resolve())+'/results2/')
    
    ldir = [n for n in os.listdir(OUTPUT_DIR) if '#' in n]
    runs = list(set([n.split('_')[0] for n in ldir]))
    return runs

def get_replications(expe):
    
    OUTPUT_DIR = pathlib.Path(str(pathlib.Path('../..').resolve())+'/results2/')
    
    runs = [n for n in os.listdir(OUTPUT_DIR) if expe in n]
    return runs


def get_expe_results(run):
    
    OUTPUT_DIR = pathlib.Path(str(pathlib.Path('../..').resolve())+'/results2/')

    repls = get_replications(run)
    
    res = {}
    n_repl = 0
    for repl in repls:
    
        path = str(OUTPUT_DIR)+'/'+repl
        df_plan = pd.read_feather(path+'/'+'expe_plan.feather')
        print(df_plan)

        res_i = {}
        for i,r in df_plan.iterrows():

            try:
                preds = pd.read_feather(path+'/'+r.run_suffix+'_PREDS.feather')
                
                competitors = ['scores']
                
                res_i[str(r.run_suffix)] = {c: preds.values.mean() for c in competitors}

            except:
                print("Couldn't open results for "+str(r.run_suffix))
    
        res_i = pd.DataFrame(res_i).T
        df_plan = df_plan.set_index('run_suffix')
        df = pd.concat((df_plan,res_i), axis=1)

        df = df.melt(id_vars=['pool','dataset','k_init','k_refinement','max_epochs','k_neighbors', 'n_sampling'], 
            var_name="competitor", 
            value_name="f1_score")
        
        df['n_replication'] = n_repl

        res[n_repl] = df        
        n_repl += 1
        
    res = pd.concat(res, axis=0).reset_index(drop=True)
        
    return res