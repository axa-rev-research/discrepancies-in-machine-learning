import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score


import sys, os
sys.path.append(os.path.dirname(sys.path[0]))


from tqdm import tqdm


def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    print("odds", odds)
    return np.min([odds, 1/odds]) * 100



### BASE CLASSES: BASIC CLASSIFIER AND FAIR CLASSIFIER


class Classifier(nn.Module):
    '''
    Generic fully-connected classifier. 
    
    n_features: X_train.shape[1]
    n_hlayers: number of hidden layers
    n_hidden: number of neurons per hidden layer
    [N.B.: architecture definition should be more flexible...]
    
    '''
    def __init__(self, n_features, n_hlayers=5, n_hidden=30):
        super(Classifier, self).__init__()
        
        architecture = [nn.Linear(n_features, n_hidden), nn.ReLU()] ## TODO : not practical right now...
        for i in range(n_hlayers):  
            architecture.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        architecture.append(nn.Linear(n_hidden, 1))    
        
        self.network = nn.Sequential(
            *architecture
            )
        
        self.n_hlayers = n_hlayers
        self.n_hidden = n_hidden
        
        
    def forward(self, x, logits=False):
        if logits == True:
            p_logits = self.network(x)
            return torch.sigmoid(self.network(x)), p_logits
        else:    
            return torch.sigmoid(self.network(x))

    
    
    def _optimizer(self):
        return optim.Adam(self.parameters(), lr=0.0001)
    
    
    def fit(self, X, y, n_epochs=10, batch_size=64, plot=False):
        
        dataset_train = TensorDataset(torch.Tensor(X.values), torch.Tensor(y.values).reshape(-1, 1))
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        
        loss_criterion = nn.BCELoss()
        optimizer = self._optimizer()
    
        loss_evolution = []
        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0
            
            for x, y in train_loader:
                optimizer.zero_grad() # clear gradients
                p_y = self(x.float())
                loss_x = loss_criterion(p_y, y)
                
                loss_x.backward()
                optimizer.step()
                
                epoch_loss += loss_x.item()
            loss_evolution.append(epoch_loss)
            
        if plot == True:
            sns.lineplot(data=pd.DataFrame(loss_evolution))
            plt.title(label='BCE loss')
            plt.xlabel("iterations")
            plt.xticks(range(n_epochs))
            plt.show()

    def predict(self, X):
        '''
        Just to normalize usage with other classifiers
        '''
        dataset_test = TensorDataset(torch.Tensor(X.values), torch.Tensor(np.ones(X.shape)))
        test_loader = DataLoader(dataset_test, batch_size=None, shuffle=False, drop_last=False)
        
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                preds.append(self(x).detach().numpy()[0])
        return (np.array(preds) >= 0.5).astype('int')
    
    def predict_proba(self, X):
        dataset_test = TensorDataset(torch.Tensor(X.values), torch.Tensor(np.ones(X.shape)))
        test_loader = DataLoader(dataset_test, batch_size=None, shuffle=False, drop_last=False)
        
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                preds.append(self(x).detach().numpy()[0])
        return np.array(preds)
    
    def _eval(self, X, Z, y):
        pred_y = self.predict(X)
        prob_y = self.predict_proba(X)
        
        acc = accuracy_score(y, pred_y)
        auc = roc_auc_score(y, prob_y)
        
        _, _, _, _, _, _, demo_parity, _, _, equ_odds = get_fairness_metrics(y, pred_y, Z, 1 - Z, 0.5)
        
        return {'Accuracy': acc, 'AUC': auc, 'Demographic parity': demo_parity, 'Equalized odds': equ_odds}
        

            

class Adversarial(nn.Module):
    '''
    Adversarial classifier, taking as input a 1-D vector of prediction (output of a Classifier.predict(logits=True)) and trying to predict a sensitive variable 
    
    n_hlayers: number of hidden layers
    n_hidden: number of neurons per hidden layer
    
    '''
    def __init__(self, n_hlayers=5, n_hidden=10):
        super(Adversarial, self).__init__()
        
        architecture = [nn.Linear(1, n_hidden), nn.ReLU()]
        for i in range(n_hlayers):  
            architecture.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        architecture.append(nn.Linear(n_hidden, 1))    
        
        self.network = nn.Sequential(
            *architecture
            )
        
        self.n_hlayers = n_hlayers
        self.n_hidden = n_hidden
        
        
    def forward(self, x, logits=False):
        #return self.network(x)
        return torch.sigmoid(self.network(x))

    
    
    def _optimizer(self):
        return optim.Adam(self.parameters(), lr=0.01)
            


class FairZhang:
    '''
    Fair classifier (kind of) based on Zhang et al. 2019. Adversarial training to make p_y independant of the sensitive attribute
    '''

    def __init__(self, clf, lambda_=1.0, adversarial_architecture=[2,5]):

        self.clf = clf # clf is used as input to ensure fair comparison with biased classifier
        self.adv = Adversarial(*adversarial_architecture)
        
        self.lambda_ = lambda_
                          
        self.clf_criterion = nn.BCELoss()
        self.clf_optimizer = self.clf._optimizer()
        self.adv_criterion = nn.BCELoss()
        self.adv_optimizer = self.adv._optimizer()

    def _optimizer(self, model):
        return optim.Adam(model.parameters())
                          
    def pretrain_both(self, X, Z, y, n_clf_epochs=10, n_adv_epochs=10, batch_size=64):
        '''
        Need to pretrain clf and adversarial if not trained manually
        '''
        dataset_train = TensorDataset(torch.Tensor(X.values), torch.Tensor(Z.values).reshape(-1, 1), torch.Tensor(y.values).reshape(-1, 1))
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True)
        
        for _ in range(n_clf_epochs):
            self._pretrain_classifier(train_loader)
        for _ in range(n_adv_epochs):
            self._pretrain_adversarial(train_loader)    
    
    def _pretrain_classifier(self, data_loader):
        for x, _, y in data_loader:
            p_y = self.clf(x)
            loss = self.clf_criterion(p_y, y)
            self.clf.zero_grad()
            loss.backward()
            self.clf_optimizer.step()

    def _pretrain_adversarial(self, data_loader):
        for x, z, _ in data_loader:
            logits_y = self.clf.forward(x, logits=True)[1].detach()
            p_z = self.adv(logits_y)
            loss = self.adv_criterion(p_z, z)
            self.adv.zero_grad()
            loss.backward()
            self.adv_optimizer.step()

    
    def predict_sensitive_adv(self, X):
        '''
        yes
        '''
        dataset_test = TensorDataset(torch.Tensor(X.values), torch.Tensor(np.ones(X.shape)))
        test_loader = DataLoader(dataset_test, batch_size=None, shuffle=False, drop_last=False)
        
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                logits_y = self.clf.forward(x, logits=True)[1]
                pred_z = self.adv(logits_y)
                preds.append(pred_z.detach().numpy())
        return np.array(preds)

    
    def fit(self, X, Z, Y, n_epochs=10, batch_size=64, plot=False):
        dataset_train = TensorDataset(torch.Tensor(X.values), torch.Tensor(Z.values).reshape(-1, 1), torch.Tensor(Y.values).reshape(-1, 1))
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        
        '''self.clf_criterion = nn.BCELoss()
        self.clf_optimizer = self._optimizer(self.clf)
        #adv_criterion = nn.BCELoss()
        self.adv_criterion = nn.MSELoss()
        self.adv_optimizer = self._optimizer(self.adv)
        #faut il initialiser optimizer 2 FOIS??''' # TODO
        
        '''self.clf_criterion = nn.BCELoss()
        self.clf_optimizer = self.clf._optimizer()
        self.adv_criterion = nn.BCELoss()
        self.adv_optimizer = self.adv._optimizer()'''
        
        prediction_loss, adversarial_loss, total_loss, clf_accuracy, adv_accuracy = [], [], [], [], []
        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0
            
            for x, z, y in train_loader:    
                logits_y = self.clf.forward(x, logits=True)[1].float()
                p_z = self.adv(logits_y)
                loss_adv_x = self.adv_criterion(p_z, z)
                
                self.adv_optimizer.zero_grad()
                loss_adv_x.backward()
                self.adv_optimizer.step()
            
            for x, z, y, in train_loader: # the idea (taken from code found somewhere) is that the adversary is trained on the full dataset while the classifier only gets a single batch of data to correct. TODO: check if relevant...
                pass
            
            
            p_y, logits_y = self.clf.forward(x, logits=True)
            p_z = self.adv(logits_y)    
            loss_adv = self.adv_criterion(p_z, z)
            pred_loss = self.clf_criterion(p_y, y)
            clf_loss = pred_loss - self.lambda_ * loss_adv
            
            self.clf_optimizer.zero_grad()
            clf_loss.backward()
            self.clf_optimizer.step()
            
            #evaluation at current epoch
            adv_loss_all, pred_loss_all, total_loss_all, clf_accuracy_all, adv_accuracy_all = self._get_current_loss(X, Z, Y)
            total_loss.append(total_loss_all)
            prediction_loss.append(pred_loss_all)
            adversarial_loss.append(adv_loss_all)
            clf_accuracy.append(clf_accuracy_all)
            adv_accuracy.append(adv_accuracy_all)
            
            
        if plot == True:
            df = pd.DataFrame({"Prediction loss": prediction_loss, "Adversarial loss": adversarial_loss})
            #print(df)
            #sns.lineplot(data=df) #TODO fix seaborn bug
            plt.plot(df)
            plt.title(label='BCE loss')
            plt.xlabel("iterations")
            plt.xticks(range(n_epochs))
            plt.show()
            
            plt.plot(total_loss)
            plt.title(label='Total loss')
            plt.xlabel("iterations")
            plt.xticks(range(n_epochs))
            plt.show()
            
            df_acc = pd.DataFrame({"Clf accuracy": clf_accuracy, "Adversarial accuracy": adv_accuracy})
            plt.plot(df_acc)
            plt.title(label='Accuracy of clf and adv')
            plt.xlabel("iterations")
            plt.xticks(range(n_epochs))
            plt.show()
        
        #return self.clf, self.adv

    def _eval(self, X, Z, y):
        pred_y = self.clf.predict(X)
        prob_y = self.clf.predict_proba(X)
        
        acc = accuracy_score(y, pred_y)
        auc = roc_auc_score(y, prob_y)
        
        _, _, _, _, _, _, demo_parity, _, _, equ_odds = get_fairness_metrics(y, pred_y, Z, 1 - Z, 0.5)
        
        return {'Accuracy': acc, 'AUC': auc, 'Demographic parity': demo_parity, 'Equalized odds': equ_odds}
    
    
    
    def _get_current_loss(self, X, Z, Y):
        '''
        get loss of current clf and adv for current epoch, over the whole training set
        '''
        dataset_data = TensorDataset(torch.Tensor(X.values), torch.Tensor(Z.values).reshape(-1, 1), torch.Tensor(Y.values).reshape(-1, 1))
        data_loader = DataLoader(dataset_data, batch_size=32, shuffle=False, drop_last=False)
        
        with torch.no_grad():
            loss_adv, pred_loss, clf_loss, clf_accuracy, adv_accuracy = 0, 0, 0, 0, 0
            for x, z, y in data_loader:
                p_y, logits_y = self.clf.forward(x, logits=True)
                p_z = self.adv(logits_y)    
                loss_adv += self.adv_criterion(p_z, z)
                pred_loss += self.clf_criterion(p_y, y)
                clf_loss += (pred_loss - self.lambda_ * loss_adv)
                clf_accuracy = accuracy_score(y, np.squeeze(p_y)>0.5)
                adv_accuracy = accuracy_score(z, np.squeeze(p_z)>0.5)
                
        return loss_adv, pred_loss, clf_loss, clf_accuracy, adv_accuracy
    
    
class FairZhang2:
    '''
    test with training on whole dataset for both
    '''

    def __init__(self, clf, lambda_=1.0, adversarial_architecture=[2,5]):

        self.clf = clf # clf is used as input to ensure fair comparison with biased classifier
        self.adv = Adversarial(*adversarial_architecture)
        
        self.lambda_ = lambda_
                          
        self.clf_criterion = nn.BCELoss()
        self.clf_optimizer = self.clf._optimizer()
        self.adv_criterion = nn.BCELoss()
        self.adv_optimizer = self.adv._optimizer()

    def _optimizer(self, model):
        return optim.Adam(model.parameters())
                          
    def pretrain_both(self, X, Z, y, n_clf_epochs=10, n_adv_epochs=10, batch_size=64):
        '''
        Need to pretrain clf and adversarial if not trained manually
        '''
        dataset_train = TensorDataset(torch.Tensor(X.values), torch.Tensor(Z.values).reshape(-1, 1), torch.Tensor(y.values).reshape(-1, 1))
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True)
        
        for _ in range(n_clf_epochs):
            self._pretrain_classifier(train_loader)
        for _ in range(n_adv_epochs):
            self._pretrain_adversarial(train_loader)    
    
    def _pretrain_classifier(self, data_loader):
        for x, _, y in data_loader:
            p_y = self.clf(x)
            loss = self.clf_criterion(p_y, y)
            self.clf.zero_grad()
            loss.backward()
            self.clf_optimizer.step()

    def _pretrain_adversarial(self, data_loader):
        for x, z, _ in data_loader:
            logits_y = self.clf.forward(x, logits=True)[1].detach()
            p_z = self.adv(logits_y)
            loss = self.adv_criterion(p_z, z)
            self.adv.zero_grad()
            loss.backward()
            self.adv_optimizer.step()

    
    def predict_sensitive_adv(self, X):
        '''
        yes
        '''
        dataset_test = TensorDataset(torch.Tensor(X.values), torch.Tensor(np.ones(X.shape)))
        test_loader = DataLoader(dataset_test, batch_size=None, shuffle=False, drop_last=False)
        
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                logits_y = self.clf.forward(x, logits=True)[1]
                pred_z = self.adv(logits_y)
                preds.append(pred_z.detach().numpy())
        return np.array(preds)

    
    def fit(self, X, Z, Y, n_epochs=10, batch_size=64, plot=False):
        dataset_train = TensorDataset(torch.Tensor(X.values), torch.Tensor(Z.values).reshape(-1, 1), torch.Tensor(Y.values).reshape(-1, 1))
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        
        '''self.clf_criterion = nn.BCELoss()
        self.clf_optimizer = self._optimizer(self.clf)
        #adv_criterion = nn.BCELoss()
        self.adv_criterion = nn.MSELoss()
        self.adv_optimizer = self._optimizer(self.adv)
        #faut il initialiser optimizer 2 FOIS??''' # TODO
        
        '''self.clf_criterion = nn.BCELoss()
        self.clf_optimizer = self.clf._optimizer()
        self.adv_criterion = nn.BCELoss()
        self.adv_optimizer = self.adv._optimizer()'''
        
        prediction_loss, adversarial_loss, total_loss, clf_accuracy, adv_accuracy = [], [], [], [], []
        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0
            
            for x, z, y in train_loader:    
                logits_y = self.clf.forward(x, logits=True)[1].float()
                p_z = self.adv(logits_y)
                loss_adv_x = self.adv_criterion(p_z, z)
                
                self.adv_optimizer.zero_grad()
                loss_adv_x.backward()
                self.adv_optimizer.step()
            
            for x, z, y, in train_loader: # change happens here compared to FairZhang()                            
                p_y, logits_y = self.clf.forward(x, logits=True)
                p_z = self.adv(logits_y)    
                loss_adv = self.adv_criterion(p_z, z)
                pred_loss = self.clf_criterion(p_y, y)
                clf_loss = pred_loss - self.lambda_ * loss_adv
            
                self.clf_optimizer.zero_grad()
                clf_loss.backward()
                self.clf_optimizer.step()
            
            #evaluation at current epoch
            adv_loss_all, pred_loss_all, total_loss_all, clf_accuracy_all, adv_accuracy_all = self._get_current_loss(X, Z, Y)
            total_loss.append(total_loss_all)
            prediction_loss.append(pred_loss_all)
            adversarial_loss.append(adv_loss_all)
            clf_accuracy.append(clf_accuracy_all)
            adv_accuracy.append(adv_accuracy_all)
            
            
        if plot == True:
            df = pd.DataFrame({"Prediction loss": prediction_loss, "Adversarial loss": adversarial_loss})
            #print(df)
            #sns.lineplot(data=df) #TODO fix seaborn bug
            plt.plot(df)
            plt.title(label='BCE loss')
            plt.xlabel("iterations")
            plt.xticks(range(n_epochs))
            plt.show()
            
            plt.plot(total_loss)
            plt.title(label='Total loss')
            plt.xlabel("iterations")
            plt.xticks(range(n_epochs))
            plt.show()
            
            df_acc = pd.DataFrame({"Clf accuracy": clf_accuracy, "Adversarial accuracy": adv_accuracy})
            plt.plot(df_acc)
            plt.title(label='Accuracy of clf and adv')
            plt.xlabel("iterations")
            plt.xticks(range(n_epochs))
            plt.show()
        
        #return self.clf, self.adv

    def _eval(self, X, Z, y):
        pred_y = self.clf.predict(X)
        prob_y = self.clf.predict_proba(X)
        
        acc = accuracy_score(y, pred_y)
        auc = roc_auc_score(y, prob_y)
        
        _, _, _, _, _, _, demo_parity, _, _, equ_odds = get_fairness_metrics(y, pred_y, Z, 1 - Z, 0.5)
        
        return {'Accuracy': acc, 'AUC': auc, 'Demographic parity': demo_parity, 'Equalized odds': equ_odds}
    
    
    
    def _get_current_loss(self, X, Z, Y):
        '''
        get loss of current clf and adv for current epoch, over the whole training set
        '''
        dataset_data = TensorDataset(torch.Tensor(X.values), torch.Tensor(Z.values).reshape(-1, 1), torch.Tensor(Y.values).reshape(-1, 1))
        data_loader = DataLoader(dataset_data, batch_size=32, shuffle=False, drop_last=False)
        
        with torch.no_grad():
            loss_adv, pred_loss, clf_loss, clf_accuracy, adv_accuracy = 0, 0, 0, 0, 0
            for x, z, y in data_loader:
                p_y, logits_y = self.clf.forward(x, logits=True)
                p_z = self.adv(logits_y)    
                loss_adv += self.adv_criterion(p_z, z)
                pred_loss += self.clf_criterion(p_y, y)
                clf_loss += (pred_loss - self.lambda_ * loss_adv)
                clf_accuracy = accuracy_score(y, np.squeeze(p_y)>0.5)
                adv_accuracy = accuracy_score(z, np.squeeze(p_z)>0.5)
                
        return loss_adv, pred_loss, clf_loss, clf_accuracy, adv_accuracy
    
    
        
    
### BUILD POOL (inspired from pool.py

class ZhangPool():
    '''
    Basic DNN pool object: gets a list of classifiers and trains them + defines new attributes
    
    
    '''
    
    def __init__(self, lambda_=1.0):
        self.lambda_ = lambda_
        
    def fit(self, X, Z, y, n_epochs=10, plot=False, n_epochs_clf=10, n_epochs_adv=10):
        
        clf_biased, clf_dummy = get_fair_pool(X.shape[1])
        clf_fair = FairZhang2(clf_dummy, lambda_=self.lambda_) #FairZhang ou FairZhang2, attention
        
        clf_biased.fit(X, y, n_epochs=n_epochs, plot=plot)
        clf_fair.pretrain_both(X, Z, y, n_clf_epochs=n_epochs_clf, n_adv_epochs=n_epochs_adv)
        clf_fair.fit(X, Z, y, n_epochs=n_epochs, plot=plot)
        
        self.models = [clf_biased, clf_fair.clf]
        
        
        #self._eval(clf_biased, clf_fair, X, Z, y)
        
        return self
    
    
    
    def _eval(self, X, Z, y):
        clf_b, clf_f = self.models
        return  clf_b._eval(), clf_f._eval()
    
    def predict(self, X, mode='classification'): #mode is useless but required for p2g
        
        dataset_test = TensorDataset(torch.Tensor(pd.DataFrame(X).values), torch.Tensor(np.ones(X.shape)))
        test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False, drop_last=False)

        preds = {}
        with torch.no_grad():
            for p in range(len(self.models)):
                pred_x = []
                for x, _ in test_loader:
                    p_y = self.models[p](x)
                    labels = (p_y >= 0.5).int().detach().numpy()
                    pred_x.extend(labels.flatten())
                preds[p] = pred_x
        preds = pd.DataFrame(preds)

        return preds
    
    
    
    def predict_proba(self, X, target=0):
        return 1
    
    
    
    def predict_discrepancies(self, X):
        """
        return 0 if no discrepancy between classifier for the prediction, return 1 if there are discrepancies
        """
        preds = self.predict(X)
        preds = preds.nunique(axis=1)
        return (preds>1).astype(int)
    
    
    def predict_mode(self, X):
        preds = self.predict(X)
        return preds.mode(axis=1)
    
    
    
    
            
### OTHER CLASSES: DEFAULT POOLS
            

def get_default_pool(n_features):
    '''
    A default DNN pool to go fast
    '''
    clf1 = Classifier(n_features,
                      n_hlayers=3,
                      n_hidden=20),
    clf2 = Classifier(n_features,
                      n_hlayers=10,
                      n_hidden=10)
    
    model_pool = [clf1, clf2]
    return model_pool


def get_fair_pool(n_features):
    '''
    A default DNN pool with fair and biased classifier
    '''
    clf_biased = Classifier(n_features,
                            n_hlayers=5,
                            n_hidden=20)
    clf_dummy = Classifier(n_features, 
                           n_hlayers=5,
                            n_hidden=20)
    
    return [clf_biased, clf_dummy]



    
def get_fairness_metrics(actual_labels, y_pred, protected_labels, non_protected_labels, thres):
    def get_toxicity_rates(y_pred, protected_labels, non_protected_labels, thres):
        protected_ops = y_pred[protected_labels == 1]
        protected_prob = sum(protected_ops)/len(protected_ops)

        non_protected_ops = y_pred[non_protected_labels == 1]
        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

        return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

    def get_true_positive_rates(actual_labels, y_pred, protected_labels, non_protected_labels, thres):

        protected_ops = y_pred[np.bitwise_and(protected_labels == 1, actual_labels == 1)]
        protected_prob = sum(protected_ops)/len(protected_ops)

        non_protected_ops = y_pred[np.bitwise_and(non_protected_labels == 1, actual_labels == 1)]
        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

        return np.round(protected_prob, 2), np.round(non_protected_prob, 2)


    def get_false_positive_rates(actual_labels, y_pred, protected_labels, non_protected_labels, thres):

        protected_ops = y_pred[np.bitwise_and(protected_labels == 1, actual_labels ==0)]
        protected_prob = sum(protected_ops)/len(protected_ops)

        non_protected_ops = y_pred[np.bitwise_and(non_protected_labels == 1, actual_labels == 0)]
        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

        return np.round(protected_prob, 2), np.round(non_protected_prob, 2)

    def demographic_parity(y_pred, protected_labels, non_protected_labels, thres):

        protected_ops = y_pred[protected_labels == 1]
        protected_prob = sum(protected_ops)/len(protected_ops)

        non_protected_ops = y_pred[non_protected_labels == 1]
        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

        return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male

  # | P_female(C = 1| Y = 1) - P_male(C = 1 | Y = 1) | < thres
    def true_positive_parity(actual_labels, y_pred, protected_labels, non_protected_labels, thres):

        protected_ops = y_pred[np.bitwise_and(protected_labels == 1, actual_labels == 1)]
        protected_prob = sum(protected_ops)/len(protected_ops)

        non_protected_ops = y_pred[np.bitwise_and(non_protected_labels == 1, actual_labels == 1)]
        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

        return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male

  # | P_female(C = 1| Y = 0) - P_male(C = 1 | Y = 0) | < thres
    def false_positive_parity(actual_labels, y_pred, protected_labels, non_protected_labels, thres):

        protected_ops = y_pred[np.bitwise_and(protected_labels == 1, actual_labels ==0)]
        protected_prob = sum(protected_ops)/len(protected_ops)

        non_protected_ops = y_pred[np.bitwise_and(non_protected_labels == 1, actual_labels == 0)]
        non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)

        return abs(protected_prob - non_protected_prob) #later take absolute diff - but we want to show females predicted more toxic than male


  # Satisfy both true positive parity and false positive parity
    def equalized_odds(actual_labels, y_pred, protected_labels, non_protected_labels, thres):
        return true_positive_parity(actual_labels, y_pred, protected_labels, non_protected_labels, thres) + false_positive_parity(actual_labels, y_pred, protected_labels, non_protected_labels, thres)

    female_tox_rate, nf_tox_rate = get_toxicity_rates(y_pred, protected_labels, non_protected_labels, thres)
    female_tp_rate, nf_tp_rate = get_true_positive_rates(actual_labels, y_pred, protected_labels, non_protected_labels, thres)
    female_fp_rate, nf_fp_rate = get_false_positive_rates(actual_labels, y_pred, protected_labels, non_protected_labels, thres)
    demo_parity = demographic_parity(y_pred, protected_labels, non_protected_labels, thres)
    tp_parity = true_positive_parity(actual_labels, y_pred, protected_labels, non_protected_labels, thres)
    fp_parity = false_positive_parity(actual_labels, y_pred, protected_labels, non_protected_labels, thres)
    equ_odds = equalized_odds(actual_labels, y_pred, protected_labels, non_protected_labels, thres)

    return female_tox_rate, nf_tox_rate, female_tp_rate, nf_tp_rate, female_fp_rate, nf_fp_rate, demo_parity, tp_parity, fp_parity, equ_odds

        
        
        
        
        