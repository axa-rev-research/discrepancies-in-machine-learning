
import pandas as pd
import numpy as np

#from autosklearn.classification import AutoSklearnClassifier
#from autogluon import TabularPrediction as task

#from tpot import TPOTClassifier


from sklearn.base import BaseEstimator, ClassifierMixin

import sklearn.datasets
import sklearn.svm
import sklearn.ensemble
import sklearn.neighbors
import sklearn.tree
import sklearn.metrics
import xgboost as xgb
import sklearn.linear_model

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

    def agreement(self, X):
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
    def __init__(self, models=['SVMrbf', 
                               #'SVMsigmoid'#,
                               'GB',
                               'XGB',
                               'LR',
                               'RF200', 'RF100', 
                               'KNN5'
                                ], RANDOM_STATE=None):
        
        self._model_types = models
        self.RANDOM_STATE = RANDOM_STATE

    def fit(self, X, y):
        """
        X: pd.DataFrame, training set's input
        y: pd.DataFrame, training set's target
        """

        self.models = {}

        if 'SVMrbf' in self._model_types:
            clf = sklearn.svm.SVC(kernel='rbf', probability=True, random_state=self.RANDOM_STATE)
            clf.fit(X,y)
            self.models['SVMrbf'] = clf

        if 'SVMpoly' in self._model_types:
            clf = sklearn.svm.SVC(kernel='poly', random_state=self.RANDOM_STATE)
            clf.fit(X,y)
            self.models['SVMpoly'] = clf

        if 'SVMsigmoid' in self._model_types: 
            clf = sklearn.svm.SVC(kernel='sigmoid', random_state=self.RANDOM_STATE)
            clf.fit(X,y)
            self.models['SVMsigmoid'] = clf

        if 'RF50' in self._model_types:
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators=50, random_state=self.RANDOM_STATE)
            clf.fit(X,y)
            self.models['RF50'] = clf
            
        if 'RF100' in self._model_types:
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=3, random_state=self.RANDOM_STATE)
            clf.fit(X,y)
            self.models['RF100'] = clf
            
        if 'RF200' in self._model_types:
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=10, random_state=self.RANDOM_STATE)
            clf.fit(X,y)
            self.models['RF200'] = clf
            
        if 'KNN5' in self._model_types:
            clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=20)
            clf.fit(X,y)
            self.models['KNN5'] = clf
        
        if 'XGB' in self._model_types:
            clf = xgb.XGBClassifier(max_depth=10, random_state=self.RANDOM_STATE)
            clf.fit(X,y)
            self.models['XGB'] = clf
            
        if 'LR' in self._model_types:
            clf = sklearn.linear_model.LogisticRegression(random_state=self.RANDOM_STATE)
            clf.fit(X,y)
            self.models['LR'] = clf
            
        if 'GB' in self._model_types:
            clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=200, random_state=self.RANDOM_STATE)
            clf.fit(X,y)
            self.models['GB'] = clf
            
        if 'DT' in self._model_types:
            clf = sklearn.tree.DecisionTreeClassifier(min_samples_leaf=5, random_state=self.RANDOM_STATE)
            clf.fit(X,y)
            self.models['DT'] = clf

        return self

    def predict(self, X, mode='classification'):

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
        preds = (preds>1).astype(int)
        preds.name = 'discrepancies'
        return preds

    def predict_mode(self, X):
        preds = self.predict(X)
        return preds.mode(axis=1)
    
    def get_performances(self, X, y):

        preds = self.predict(X)
        f1_scores = {}
        for c in self.models:
            f1_scores[c] = sklearn.metrics.f1_score(y, preds[c])
        return f1_scores
    
    def filter_accuracies(self, X, y, max_delta_accuracies=1000.0):
        
        try:
            f1_scores = self.get_performances(X, y)
            best_ = max(f1_scores.values())
            models_tokeep_ = [k for k, v in f1_scores.items() if v >= best_ - max_delta_accuracies]
            self.models = {k:self.models[k] for k in models_tokeep_}
        except AttributeError:
            print("Pool must be trained before filtering.")
        return self

    
class BasicPool2(Pool):
    def __init__(self, models=[#'RF1', 'RF2', 
                               'RF3', 'RF4', 'RF5'
                                #'XGB1','XGB2','XGB3','XGB4'
                                ]):
        
        self._model_types = models


    def fit(self, X, y):
        """
        X: pd.DataFrame, training set's input
        y: pd.DataFrame, training set's target
        """

        self.models = {}
        if 'RF1' in self._model_types:
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=5, criterion='gini')
            clf.fit(X,y)
            self.models['RF1'] = clf
        
        if 'RF2' in self._model_types:
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=4, criterion='gini')
            clf.fit(X,y)
            self.models['RF2'] = clf
        
        if 'RF3' in self._model_types:
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=20, criterion='gini')
            clf.fit(X,y)
            self.models['RF3'] = clf
            
        if 'RF4' in self._model_types:
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=None, criterion='gini')
            clf.fit(X,y)
            self.models['RF4'] = clf
            
        if 'RF5' in self._model_types:
            clf = sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=10, criterion='gini')
            clf.fit(X,y)
            self.models['RF5'] = clf
        
        
        
        ####
            
        if 'XGB1' in self._model_types:
            clf = xgb.XGBClassifier(max_depth=3)
            clf.fit(X,y)
            self.models['XGB1'] = clf
            
        if 'XGB2' in self._model_types:
            clf = xgb.XGBClassifier(max_depth=5)
            clf.fit(X,y)
            self.models['XGB2'] = clf
            
        if 'XGB3' in self._model_types:
            clf = xgb.XGBClassifier(max_depth=10)
            clf.fit(X,y)
            self.models['XGB3'] = clf
            
        if 'XGB4' in self._model_types:
            clf = xgb.XGBClassifier(max_depth=None)
            clf.fit(X,y)
            self.models['XGB4'] = clf
            
        return self

    def predict(self, X, mode='classification'):

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
        preds = (preds>1).astype(int)
        preds.name = 'discrepancies'
        return preds


    def predict_mode(self, X):
        preds = self.predict(X)
        return preds.mode(axis=1)
    
    def get_performances(self, X, y):

        preds = self.predict(X)
        f1_scores = {}
        for c in self.models:
            f1_scores[c] = sklearn.metrics.f1_score(y, preds[c])
        return f1_scores

    

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
        preds =  (preds>1).astype(int)
        preds.name = 'discrepancies'
        return preds


    def predict_mode(self, X):
        preds = self.predict(X)
        return preds.mode(axis=1)


class AutogluonPool(Pool):

    def __init__(self, max_delta_accuracies=0.05):
        self.max_delta_accuracies = max_delta_accuracies


    def fit(self, X, y, output_directory, time_limit=100):

        self.X_columns = X.columns.to_list()
        train_data = self.get_df_4_autogluon(X,y)

        self.predictor = task.fit(train_data=train_data, time_limits=time_limit, label='class', verbosity=0, output_directory=output_directory)
        
        # introducing delta max accuracies for autogluon
        test_data = self.get_df_4_autogluon(X, y)
        leaderboard = self.predictor.leaderboard(test_data, silent=True)
        lowerbound_accuracy = np.max(leaderboard['score_test'])*(1-self.max_delta_accuracies)
        accuracies = leaderboard['score_test']
        self.predictor.delete_models(models_to_keep=leaderboard[leaderboard.score_test >= lowerbound_accuracy]['model'].tolist(), dry_run=False)
        
        #hotfix to align with autosklearn's nomenclature
        self.models = self.predictor.get_model_names()
        print('apres suppression')
        print(self.models)
        return self

    def predict(self, X, mode='classification', models_to_include=None):
        
        if not (isinstance(X,pd.DataFrame) or isinstance(X,task.Dataset)):
            X = pd.DataFrame(X, columns=self.X_columns)

        if mode == 'classification':

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
        preds = (preds>1).astype(int)
        preds.name = 'discrepancies'
        return preds


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
    
    
'''
class TPOTPool(Pool):

    def __init__(self, max_delta_accuracies=0.05):
        self.max_delta_accuracies = max_delta_accuracies


    def fit(self, X, y, output_directory, time_limit=100):

        self.X_columns = X.columns.to_list()
        train_data = self.get_df_4_tpot(X,y)

        self.predictor = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=40)        
        tpot.fit(train_data.iloc[:, :-1], train_data[:, -1])
        
        # introducing delta max accuracies for autogluon
        test_data = self.get_df_4_tpot(X, y)
        leaderboard = self.predictor.leaderboard(test_data, silent=True)
        lowerbound_accuracy = np.max(leaderboard['score_test'])*(1-self.max_delta_accuracies)
        accuracies = leaderboard['score_test']
        self.predictor.delete_models(models_to_keep=leaderboard[leaderboard.score_test >= lowerbound_accuracy]['model'].tolist(), dry_run=False)
        
        #hotfix to align with autosklearn's nomenclature
        self.models = self.predictor.get_model_names()
        print('apres suppression')
        print(self.models)
        return self
    
    

    def predict(self, X, mode='classification', models_to_include=None):
        
        if not (isinstance(X,pd.DataFrame) or isinstance(X,task.Dataset)):
            X = pd.DataFrame(X, columns=self.X_columns)

        if mode == 'classification':

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


    def get_df_4_tpot(self, X, y):
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

        return performance, leaderboard'''