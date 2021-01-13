
import pandas as pd
import numpy as np

from autogluon import TabularPrediction as task

import sklearn.datasets
import sklearn.svm
import sklearn.ensemble
import sklearn.tree

class Pool:

    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass

    def predict_discrepancy(self):
        pass


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
        return (preds>1)


    def predict_mode(self, X):
        preds = self.predict(X)
        return preds.mode(axis=1)


class AutogluonPool(Pool):

    def __init__(self):
        pass


    def fit(self, X, y, time_limit=2):

        self.X_columns = X.columns.to_list()
        train_data = self.get_df_4_autogluon(X,y)

        self.predictor = task.fit(train_data=train_data, time_limits=time_limit, label='class', refit_full=True, verbosity=0)

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
        return (preds>1)


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