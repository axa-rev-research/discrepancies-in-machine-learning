from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin_min, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys; sys.path.insert(0, '..') # add parent folder path where discrepancies folder is

from discrepancies import datasets, pool, pool2graph, evaluation

RANDOM_STATE = 42


class ActiveLearningExperiment:
    
    def __init__(self, dataset, split_params, graph_params, disc_filter='obs'):
        self.dataset = dataset
        self.disc_max_epochs = graph_params['max_epochs']
        self.disc_stop_criterion = graph_params['stop_criterion']
        self.train_split = split_params['train_split']
        self.pool_split = split_params['pool_split']
        self.k_init = graph_params['k_init']
        self.k_refinement = graph_params['k_refinement']
    
    
    
    def run_experiment(self, it=10):
        self.out = []
        self.accuracy_before = []
        for i in range(it):
            X_train, X_test, y_train, y_test, scaler, feature_names, target_names = datasets.get_dataset(n_samples=300,dataset=self.dataset, test_size=1 - self.train_split)
            X_eval, X_pool, y_eval, y_pool = train_test_split(X_test, y_test, test_size=self.pool_split) #oui sale
            print('Dataset sizes', X_train.shape, X_pool.shape, X_eval.shape)
            print('Class balance', y_train.mean())
            self.pool1 = pool.BasicPool().fit(X_train, y_train)
            disc_graph = pool2graph.pool2graph(X_train, y_train, self.pool1, k_init=self.k_init, k_refinement=self.k_refinement)
            disc_graph.fit(max_epochs=self.disc_max_epochs, stopping_criterion=self.disc_stop_criterion)
            X_discr, y_discr = disc_graph.get_discrepancies_dataset()
            print('Graph size', X_discr.shape)
            
            ### Accuracy baseline on train
            model_eval = ModelEval().model.fit(X_train, y_train)
            # accuracy_before.append((model_eval.predict(X_eval) == y_eval).mean())
            self.accuracy_before.append(recall_score(y_eval, model_eval.predict(X_eval)))
            
            
            ### CANDIDATES
            self.disc_filter = 'obs'            
            candidates_datasets, candidates_targets = ALStrategies(X_pool, y_pool, X_discr, y_discr, self.pool1, disc_filter=self.disc_filter).get_all_candidates()
                        
            N_COMPETITORS = len(candidates_datasets)

            
            ### ADDING LABELLED INSTANCES AND MEASURING ACCURACY
            acc_matrix_i = np.zeros((candidates_datasets[0].shape[0], N_COMPETITORS)) #out matrix for iteration i
            
            for xi in range(candidates_datasets[0].shape[0]): # on ajoute observations une par une
                
                for d in range(N_COMPETITORS): # on boucle sur les competiteurs
                    
                    new_train = np.append(X_train.copy(), candidates_datasets[d][:xi+1].reshape(xi+1, X_train.shape[1]), axis=0)
                    new_label = np.append(y_train.copy(), candidates_targets[d][:xi+1])

                    model_eval = ModelEval().model.fit(new_train, new_label)
                    
                    # acc_ixid = (model_eval.predict(X_eval) == y_eval).mean()
                    acc_ixid = recall_score(y_eval, model_eval.predict(X_eval))
                    acc_matrix_i[xi, d] = acc_ixid
            self.out.append(acc_matrix_i)
        return self
    
    
    
    
    ### STORE OR PLOT RESULTS
    def store_results(self):
        PATH = './'
        
        min_size = min([a.shape[0] for a in self.out])
        list_of_accuracy_matrix_refined = [a[:min_size, :] for a in self.out]
        big_matrix = np.dstack(list_of_accuracy_matrix_refined)
        # todo: write in csv bigmatrix and also self.accuracy_before
        return False
        
    def plot_results(self, save=False, show=True, step=1):

        min_size = min([a.shape[0] for a in self.out])
        list_of_accuracy_matrix_refined = [a[:min_size, :] for a in self.out]
        big_matrix = np.dstack(list_of_accuracy_matrix_refined)


        colors = ['blue', 'green', 'orange']
        legend_labels = ['Ours', 'Random', 'Disagreement' ]

        plt.figure(figsize=(8,5))

        for i in range(big_matrix.shape[1]):
            competitor_matrix = big_matrix[:, i, :] #size nb of obs, nb of iterations 

            mean_acc = competitor_matrix.mean(axis=1)
            std_acc = competitor_matrix.std(axis=1)
            std_acc = 1.96 * std_acc / competitor_matrix.shape[1] # intervale 95%
            mean_acc2 = []
            err = []

            for k in range(int(competitor_matrix.shape[0] / step)): # boucle pour regrouper les instances (plotter ajouts 5 par 5 au lieu de 1 par 1 par ex)
                val = mean_acc[min((k+1)*step, competitor_matrix.shape[0] - 1)]
                mean_acc2.append(float(val))
                err.append(std_acc[min((k+1)*step, competitor_matrix.shape[0] - 1)])

            ax = plt.errorbar(x=np.array(range(int(competitor_matrix.shape[0]/step)))*step, y=np.array(mean_acc2), yerr=err, fmt='-', color=colors[i], linewidth=0.8)
            ax.set_label(legend_labels[i])
        baseline = plt.hlines(np.array(self.accuracy_before).mean(), color='red', xmin=0, xmax=big_matrix.shape[0])
        baseline.set_label('No augmentation')
        plt.legend()

        if save == True:
            plt.savefig('./activelearning_%s.pdf'%self.dataset)
        if show == True:
            plt.show()
    


    
class ALStrategies:
    '''
    Apply Active Learnig strategies to generate ordered list of test instances to be added.
    Strategies: priorizing based on distance to 
    Competitors: baseline random, baseline based on pool disagreement
    
    Inputs:
    X_pool : test set to pick instances to be labelled
    y_pool : labels of this test set
    X_discr : Discrepancy dataset; obtained from P2G. Used for our strategy
    y_discr : labels of discrepancy dataset
    disc_filter : we need to know which instances from Xpool are in discrepancy: should it be "observed" using the pool, or predicted with a bbox trained on Xdiscr?
    
    Returns:
    list of new ordered instances from X_pool, list of their labels 
    '''
    def __init__(self, X_pool, y_pool, X_discr, y_discr, clf_pool, disc_filter='obs'):
        self.X = X_pool
        self.y = y_pool
        self.disc_filter = disc_filter
        self.clf_pool = clf_pool
        self.X_discr = X_discr
        self.y_discr = y_discr
        
    
    def get_all_candidates(self):
        X1, y1 = self.get_candidates_ours_closestAll(self.X_discr, self.y_discr)
        X2, y2 = self.get_candidates_baseline_random(X1.shape[0])
        X3, y3 = self.get_candidates_baseline_disagreement(X1.shape[0])        
        return [X1, X2, X3], [y1, y2, y3]
    
    
    ### STRATEGIES 
    # normalement: sortir de la classe
    ## 1. plus proches des points de discrepancy du graphe
    def get_candidates_ours_closestAll(self, X_discr, y_discr):
        if self.disc_filter == 'pred': # is discrepancy predicted with a clf or observed with the pool?
            bbox_discr = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=None).fit(X_discr, y_discr)
            is_discr = bbox_discr.predict(self.X)
        elif self.disc_filter == 'obs': #PROBLEME IMPORT ?
            is_discr = self.clf_pool.predict_discrepancies(self.X)
        else:
            raise ValueError
            
        X_test_discr = self.X.iloc[np.where(is_discr ==1)] #obs from test where discrepancy is observed/predicted
        y_test_discr = self.y.iloc[np.where(is_discr ==1)] #ligne d'après devrait être distance au centre du cluster principal?
        index_closest_all = np.argsort(
            pairwise_distances_argmin_min(X_test_discr, X_discr.iloc[np.where(y_discr == 1)])[1])
        
        X_to_add_sorted_all = np.array(X_test_discr)[index_closest_all]
        y_to_add_sorted_all = np.array(y_test_discr)[index_closest_all]
        
        return X_to_add_sorted_all, y_to_add_sorted_all
    
    ##. 2. plus proche des points du plus gros cluster
    def get_candidates_ours_closestCluster(X_discr, y_discr):
        # pas optimal là car on veut pas faire tout retourner...
        return False
    
    ## 3. Baseline random
    def get_candidates_baseline_random(self, n):
        index_random_baseline = np.random.randint(0, self.X.shape[0], n) #X_to_add_sorted_all.shape[0]
        X_to_add_baseline = np.array(self.X)[index_random_baseline]
        y_to_add_baseline = np.array(self.y)[index_random_baseline]
        return X_to_add_baseline, y_to_add_baseline
    
    ## 4. Basline disagreement
    def get_candidates_baseline_disagreement(self, n):
        index_most_disagreed = np.argsort(np.abs(self.clf_pool.predict(self.X).mean(axis=1) - 0.5))
        X_to_add_baseline = np.array(self.X)[index_most_disagreed][:n]
        y_to_add_baseline = np.array(self.y)[index_most_disagreed][:n]
        return X_to_add_baseline, y_to_add_baseline
            



class ModelEval:
    '''
    Declaring here model used to evaluate the growing accuracy of the strategies
    '''
    def __init__(self):
        # self.model = LogisticRegression(penalty='none', n_jobs=-1, random_state=None)
        # self.model = RandomForestClassifier(n_estimators=10, max_depth=4, n_jobs=1, random_state=None)
        self.model = DecisionTreeClassifier(max_depth=None)
        # self.model = SVC(kernel='rbf', random_state=None)



