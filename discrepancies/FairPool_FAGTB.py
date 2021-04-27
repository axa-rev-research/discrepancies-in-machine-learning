#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:19:13 2019

@author: vincentgrari
"""
import math 

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score




def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    print("odds", odds)
    return np.min([odds, 1/odds]) * 100


class GradientBoosting_AXA_Fair(object):

    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, max_features, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.max_features = max_features

        # Initialize regression trees
        self.trees = []
        self.clfs = []
        for _ in range(n_estimators):
            tree = DecisionTreeRegressor(criterion='friedman_mse', max_depth=9,
  max_features=self.max_features, max_leaf_nodes=None,
  min_impurity_decrease=0.0, min_impurity_split=None,
  min_samples_leaf=1, min_samples_split=2,
  min_weight_fraction_leaf=0.0,
  #presort='auto', 
   random_state=0)
            self.trees.append(tree)
            clf = LogisticRegression()           
            self.clfs.append(clf) # sert à rien à priori de garder en mémoire les adverses. 
    def lossgr(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def fit2(self, X, y, sensitive, LAMBDA):
        clf = LogisticRegression()
        clf._initialize_parameters(sensitive)
        print(clf.param)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

    def fit(self, X, y, sensitive, LAMBDA, Xtest, yt, sensitivet):

        y2 = np.expand_dims(sensitive, axis=1)

        clf = LogisticRegression()
        clf._initialize_parameters(y2)
        #for i in range(200):
        #    clf.fit(y,sensitive,1)

        self.Init = np.log(np.sum(y)/np.sum(1-y))
        #self.Init = np.mean(y)
        y_pred2 = np.full(np.shape(y), self.Init)
        y_pred = np.full(np.shape(y), self.Init)
        y_predt = np.full(np.shape(yt), self.Init)
        t =np.full(np.shape(y), 0)
        t2 =np.full(np.shape(yt), 0)
#        self.clfs.append(clf)
        self.LAMBDA = LAMBDA
        
        for i in range(self.n_estimators):
            gradient = y- 1/(1+np.exp(-y_pred)) # gradient = -1* résidus. Ici c'est juste résidus je crois 
            #gradient = self.gradient(y,y_pred)
            self.trees[i].fit(X, gradient) #hm? pq pas fit sur u aussi?
            update = self.trees[i].predict(X) #? prediction des residus

            # Update y prediction
            # Update y prediction
            
            y_pred += np.multiply(self.learning_rate, update) #predictions de F(x) updatées avec hm x. le learning rate remplace le gamma?
            y_pred2 = np.expand_dims(1/(1+np.exp(-y_pred)), axis=1)
            clf.fit(y_pred2,sensitive,1) # fit l'adversarial
            t=np.squeeze(clf.gradient_adv(y_pred2,sensitive).T) #tm ok. il y a un signe moins normalement? Inclus ?
            y_pred += - self.learning_rate*LAMBDA*t #ok mais ordre bizarre
            self.clfs[i]=clf
            y_fin = 1/(1+np.exp(-y_pred)) # pas égal à y_pred2 ?

            
            #avoir l'accurcy sur le test
            updatet = self.trees[i].predict(Xtest)
            y_predt += np.multiply(self.learning_rate, updatet) 
            y_pred2t = np.expand_dims(1/(1+np.exp(-y_predt)), axis=1)
            t2=np.squeeze(clf.gradient_adv(y_pred2t,sensitivet).T)
            y_predt2=1/(1+np.exp(-y_predt))
            y_predt += - self.learning_rate*LAMBDA*t2
            
            accuracy = accuracy_score(y, np.squeeze(y_fin)>0.5)
            accuracyt = accuracy_score(yt, np.squeeze(y_predt2)>0.5)
            if i % 20 == 0:
                print (i, ": param :", clf.param, "Accuracy:", round(accuracy,4), " test : ", round(accuracyt,4), " Prule Train : ", p_rule(y_fin, sensitive)/100," Prule test : ", p_rule(y_predt2, sensitivet)/100)
        return y_fin

    def predict(self, X, sensitive):
        y_pred = np.full(np.shape(X)[0],self.Init, self.Init)

        for i in range(self.n_estimators):
            update = self.trees[i].predict(X)
            y_pred += np.multiply(self.learning_rate, update)
            y_pred2 = np.expand_dims(1/(1+np.exp(-y_pred)), axis=1)
            t=np.squeeze(self.clfs[i].gradient_adv(y_pred2,sensitive).T)
            y_pred += - self.learning_rate*self.LAMBDA*t
            y_fin = 1/(1+np.exp(-y_pred))
            #print (self.clfs[i].param2())
        # Set label to the value that maximizes probability
        return y_fin


class GradientBoostingRegressor_AXA(GradientBoosting_AXA_Fair):
    def __init__(self, n_estimators=200, learning_rate=0.05, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4, max_features = 7, debug=False):
        super(GradientBoostingRegressor_AXA, self).__init__(n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_impurity=min_var_red,
            max_depth=max_depth,
            regression=True,
            max_features = max_features)

class GradientBoostingClassifier_AXA(GradientBoosting_AXA_Fair):
    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, max_features = 7, debug=False):
        super(GradientBoostingClassifier_AXA, self).__init__(n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_impurity=min_info_gain,
            max_depth=max_depth,
            regression=False,
            max_features = max_features)



class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class LogisticRegression():
    def __init__(self, learning_rate=.1):
        self.param = None
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))
        self.param = [0]

    def fit(self, X, y, iteration):
        y_pred = self.sigmoid(X.dot(self.param))
        self.param -= self.learning_rate * -(y - y_pred).dot(X)
        return self.param

    def gradient_adv(self,X,y):
        y_pred = self.sigmoid(X.dot(self.param))
        gradient_adv = (y - y_pred)*self.param*X.T*(1-X).T
        return gradient_adv

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred

    def lossfunction_adv(self,X,y):
        y_pred = self.sigmoid(X.dot(self.param))
        return y-y_pred

    def param2(self):
        
        return 2*self.param
    

def plot_distributions(y, Z, iteration=None, val_metrics=None, p_rules=None, fname=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    #legend={'race': ['black','white'],
    #        'sex': ['female','male']}
    legend={'sex':['non-black', 'black']}
    for idx, attr in enumerate(Z.columns):
        for attr_val in [0, 1]:
            ax = sns.distplot(y[Z[attr] == attr_val], hist=False, 
                              kde_kws={'shade': True,},
                              label='{}'.format(legend[attr][attr_val]), 
                              ax=axes[idx])
        ax.set_xlim(0,1)
        ax.set_ylim(0,15)
        #ax.set_yticks([])
        ax.set_title("sensitive attibute: {}".format(attr))
        if idx == 1:
            ax.set_ylabel('prediction distribution')
        ax.set_xlabel(r'$P({{income>50K}}|z_{{{}}})$'.format(attr))
    if iteration:
        fig.text(1.0, 0.9, f"Training iteration #{iteration}", fontsize='16')
    if val_metrics is not None:
        fig.text(1.0, 0.65, '\n'.join(["Prediction performance:",
                                       f"- ROC AUC: {val_metrics['ROC AUC']:.2f}",
                                       f"- Accuracy: {val_metrics['Accuracy']:.1f}"]),
                 fontsize='16')
    if p_rules is not None:
        fig.text(1.0, 0.4, '\n'.join(["Satisfied p%-rules:"] +
                                     [f"- {attr}: {p_rules[attr]:.0f}%-rule" 
                                      for attr in p_rules.keys()]), 
                 fontsize='16')
    fig.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
    return fig