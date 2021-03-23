
import pandas as pd
import numpy as np

from autosklearn.classification import AutoSklearnClassifier

#from autogluon import TabularPrediction as task

from sklearn.base import BaseEstimator, ClassifierMixin

import sklearn.datasets
import sklearn.svm
import sklearn.ensemble
import sklearn.tree

class Pool(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass

    def predict_discrepancies(self):
        pass

    def agreement(self, x):
        pred_agreement = {}

        i = 1
        for m in self.models:
            pred_agreement[m] = self.models[m].predict(X)
            i += 1

        pred_agreement = pd.DataFrame(pred_agreement)
        agreement = np.zeros((pred_agreement.shape[1],pred_agreement.shape[1]))

        for i in range(pred_agreement.shape[1]):
            for j in range(i+1,pred_agreement.shape[1]):
                tmp = pred_agreement.iloc[:,i]==pred_agreement.iloc[:,j]
                agreement[i,j] = tmp.sum()/tmp.shape[0]

        return agreement


class BasicPool(Pool):

    def __init__(self, models=['SVMrbf', 'SVMpoly', 'SVMsigmoid', 'RF']):
        
        self._model_types = models


    def fit(self, X, y):
        """
        X: pd.DataFrame, training set's input
        y: pd.DataFrame, training set's target
        """

        self.models = {}

        if 'SVMrbf' in self._model_types:
            clf = sklearn.svm.SVC(kernel='rbf')
            clf.fit(X,y)
            self.models['SVMrbf'] = clf

        if 'SVMpoly' in self._model_types:
            clf = sklearn.svm.SVC(kernel='poly')
            clf.fit(X,y)
            self.models['SVMpoly'] = clf

        if 'SVMsigmoid' in self._model_types: 
            clf = sklearn.svm.SVC(kernel='sigmoid')
            clf.fit(X,y)
            self.models['SVMsigmoid'] = clf

        if 'RF' in self._model_types:
            clf = sklearn.ensemble.RandomForestClassifier()
            clf.fit(X,y)
            self.models['RF'] = clf

        return self

    def predict(self, X):

        preds = {}
        for p in self.models:
            preds[p] = self.models[p].predict(X)
        preds = pd.DataFrame(preds)

        return preds

    def predict_proba(self, X, target=0):

        preds = {}
        for p in self.models:
            try:
                preds[p] = self.models[p].predict_proba(X)[:,target]
            except:
                a = np.empty((len(X),))
                a[:] = np.nan
                preds[p] = a
        preds = pd.DataFrame(preds)

        return preds

    def predict_discrepancies(self, X):
        """
        return 0 if no discrepancy between classifier for the prediction, return 1 if there are discrepancies
        """
        preds = self.predict(X)
        preds = preds.nunique(axis=1)
        # Return True if the class predicted for one instance is not unique, False if all the predictions are equal
        return (preds>1).astype(int)


    def predict_mode(self, X):
        preds = self.predict(X)
        return preds.mode(axis=1)


class AutoSklearnPool(Pool):

    def __init__(self, max_delta_accuracies=0.05, time_left_for_this_task=30, n_jobs=2):
        
        self.max_delta_accuracies = max_delta_accuracies
        self.automl = AutoSklearnClassifier(time_left_for_this_task=time_left_for_this_task, n_jobs=n_jobs)
        self.n_jobs = n_jobs
        self.time_left_for_this_task = time_left_for_this_task
        self.max_delta_accuracies = max_delta_accuracies

    def fit(self, X, y):
        """
        X: pd.DataFrame, training set's input
        y: pd.DataFrame, training set's target
        """

        self.models = {}

        self.classes_, y = np.unique(y, return_inverse=True)

        # bad hack to catch the "AttributeError: 'AutoSklearnClassifier' object has no attribute 'load_models'" error
        try:
            self.automl.fit(X, y, dataset_name='digits')
        except:
            pass

        lowerbound_accuracy = np.max(self.automl.cv_results_['mean_test_score'])*(1-self.max_delta_accuracies)
        accuracies = self.automl.cv_results_['mean_test_score']
        models_2_pool = accuracies>=lowerbound_accuracy

        self.models = {}
        j = 1
        for i, (weight, pipeline) in enumerate(self.automl.get_models_with_weights()):
            if models_2_pool[i]:
                self.models['autosklearn#'+str(j)] = pipeline
                j += 1

        return self

    def predict(self, X, mode='discrepancies'):

        if mode == 'discrepancies':
            preds = self.predict_discrepancies(X)

        elif mode == 'classification':
            preds = {}
            for p in self.models:
                preds[p] = self.models[p].predict(X)
            preds = pd.DataFrame(preds)

        return preds

    def predict_proba(self, X, target=0):

        preds = {}
        for p in self.models:
            try:
                preds[p] = self.models[p].predict_proba(X)[:,target]
            except:
                a = np.empty((len(X),))
                a[:] = np.nan
                preds[p] = a
        preds = pd.DataFrame(preds)

        return preds

    def predict_discrepancies(self, X):
        """
        return 0 if no discrepancy between classifier for the prediction, return 1 if there are discrepancies
        """
        preds = self.predict(X, mode='classification')
        preds = preds.nunique(axis=1)
        # Return True if the class predicted for one instance is not unique, False if all the predictions are equal
        return (preds>1).astype(int)


    def predict_mode(self, X):
        preds = self.predict(X)
        return preds.mode(axis=1)


class AutogluonPool(Pool):

    def __init__(self):
        pass


    def fit(self, X, y, output_directory, time_limit=2):

        self.X_columns = X.columns.to_list()
        train_data = self.get_df_4_autogluon(X,y)

        self.predictor = task.fit(train_data=train_data, time_limits=time_limit, label='class', verbosity=0, output_directory=output_directory)

        return self

    def predict(self, X, mode='individual', models_to_include=None):
        
        if not (isinstance(X,pd.DataFrame) or isinstance(X,task.Dataset)):
            X = pd.DataFrame(X, columns=self.X_columns)

        if mode == 'individual':

            preds = {}
            for p in self.predictor.get_model_names():
                if (models_to_include is None) or (p in models_to_include):
                    preds[p] = self.predictor.predict(X, model=p)
                
            preds = pd.DataFrame(preds)

        elif mode == 'autogluon':

            preds = self.predictor.predict_proba(X, as_multiclass=True)


        return preds


    def predict_discrepancies(self, X):
        """
        return 0 if no discrepancy between classifier for the prediction, return 1 if there are discrepancies
        """

        preds = self.predict(X)
        preds = preds.nunique(axis=1)
        # Return True if the class predicted for one instance is not unique, False if all the predictions are equal
        return (preds>1).astype(int)


    def get_df_4_autogluon(self, X, y):
        """
        X: pd.DataFrame, input data
        y: pd.Series, 1D target
        """

        df = pd.concat((X,y), axis=1)
        df.columns = X.columns.to_list()+['class']
        
        return df


    def get_performances(self, X, y):

        test_data = self.get_df_4_autogluon(X, y)

        performance = self.predictor.evaluate(test_data)
        leaderboard = self.predictor.leaderboard(test_data, silent=True)

        return performance, leaderboard