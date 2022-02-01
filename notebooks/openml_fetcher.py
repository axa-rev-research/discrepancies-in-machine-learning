import shelve
import requests
import logging

import numpy as np
import pandas as pd
import openml
from io import StringIO
from scipy.io.arff import loadarff
from sklearn.datasets import fetch_openml



def get_discr(task_id, epsilon=0.02, get_data=False, path='./discrepancies'):

    # Get OpenMl task attributes
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id

    dataset = openml.datasets.get_dataset(dataset_id)
    logging.warning('Fetching data for dataset: '+str(dataset.name))

    # Check if examples for the dataset was asked to be returned by user (vs. just the predictions and calculations of discrepancies)
    if get_data:
        try:
            with shelve.open(path+'/openml/openml-data', 'c') as shelf:
                data = shelf[str(dataset_id)]

            with shelve.open(path+'/openml/openml-target', 'c') as shelf:
                target = shelf[str(dataset_id)]

        except:
            (data, target) = fetch_openml(data_id=dataset_id, return_X_y=True)

            with shelve.open(path+'/openml/openml-data', 'c') as shelf:
                shelf[str(dataset_id)] = data

            with shelve.open(path+'/openml/openml-target', 'c') as shelf:
                shelf[str(dataset_id)] = target

    else:
        data, target = None, None

    # Check if we already have data in stock
    # with shelve.open(path+'/openml/openml-preds', 'c') as shelf:
    # try:
    #     with np.load(path+'/openml/openml-preds.npz') as store:
    #         preds = store[str(dataset_id)]
    #     return
    # except:
    #     pass

    # Get evaluation scores of all the runs for the given task
    # Ordered by descending predictive accuracy - only return the 100 best runs to increase perfs and limit bandwith consumption: should be sufficient given the thresholding on predictive accurancy afterwards
    eval_list = openml.evaluations.list_evaluations('predictive_accuracy', tasks=[task_id], sort_order='desc', size=100)

    # Format all the evaluation - get the evaluation score (here prediction accuracy) - format results in a pd.Series
    scores = {}
    for eval_id in eval_list:
        eval = eval_list[eval_id]
        scores[eval_id] = eval.value
    scores = pd.Series(scores)
    scores = scores.sort_values(ascending=False)

    # Select all the runs that are at most epsilon percent less performant than the best run
    threshold = scores.max()-scores.max()*epsilon
    mask = scores >= threshold
    runs_id = scores[mask].index

    # Get all the attributes of the selected runs
    runs = openml.runs.get_runs(runs_id)

    preds_dict = {}
    for run in runs:

        # Retrieve the predictions of the current run
        try:
            preds = pd.read_csv(path+'/openml/preds/'+str(run.id)+'.csv')

        except:
        
            f = requests.get(run.predictions_url)
            f = StringIO(f.text)
            raw_data = loadarff(f)
            preds = pd.DataFrame(raw_data[0])

            preds.to_csv(path+'/openml/preds/'+str(run.id)+'.csv')

        preds = preds.loc[:,['row_id','prediction']]
        preds = preds.set_index('row_id').sort_index()

        preds_dict[run.id] = preds

    preds = pd.concat(preds_dict, axis=1)

    # Store prediction accuracies
    with pd.HDFStore(path+'/openml/openml-accuracies.h5') as store:
        store[str(dataset_id)] = scores[mask]

    # Store predictions - among top runs (~predictors) on the task

    # with shelve.open(path+'/openml/openml-preds', 'c') as shelf:
    #     shelf[str(dataset_id)] = preds
    # np.savez_compressed(path+'/openml/openml-preds.npz', **{str(dataset_id):preds})
    with pd.HDFStore(path+'/openml/openml-preds.h5') as store:
        store[str(dataset_id)] = preds

    # Is there prediction discrepancies (ie. more than one different prediction per example)
    tmp = preds.nunique(axis=1)
    discr = tmp!=1
    discr = discr.astype('int')

    # Store discrepancies - among top runs (~predictors) on the task

    # with shelve.open(path+'/openml/openml-discr', 'c') as shelf:
    #     shelf[str(dataset_id)] = discr
    # np.savez_compressed(path+'/openml/openml-discr.npz', **{str(dataset_id):discr})
    with pd.HDFStore(path+'/openml/openml-discr.h5') as store:
        store[str(dataset_id)] = discr

    logging.warning(discr.sum()/discr.shape[0])