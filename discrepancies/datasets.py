import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_wine, fetch_20newsgroups_vectorized, fetch_kddcup99, make_moons, fetch_openml

from sklearn.manifold import *


RANDOM_STATE = 42

def get_dataset(dataset='half-moons',
                n_samples=1000,
                test_size=0.33,
                standardize=True,
                noise=0.3):
    """
    dataset: str, name of the dataset to load in {'breast-cancer', 'half-moons'}
    n_samples: int, number of instances to return in total (train+test). For synthetic datasets, number of generated instances. For real datasets, number of randomly drawn instances.
    test_size: float, size of the test set (in [0,1]).
    standardize: bool, defines if returned dataset is standardized
    noise: for synthetic dataset, when applicable (e.g. half moons), defines the amount of noise in the generated data

    List of datasets:

    Used in TabNet:
    - Forest Cover Type
    - Poker Hand
    - Sarcos Robotics Arm Inverse Dynamics
    - Higgs Boson
    - Rossmann Store Sales
    - KDD datasets


    """
    
    cat_names = []

    if dataset == 'breast-cancer':
        data = load_breast_cancer(return_X_y=False)
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names

    elif dataset == 'load-wine':
        data = load_wine(return_X_y=False)
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names

    elif dataset == 'half-moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=RANDOM_STATE)
        feature_names = [str(i) for i in range(X.shape[1])]
        target_names = [str(i) for i in np.unique(y)]

    elif dataset == 'kddcup99':
        data = fetch_kddcup99(subset=None, return_X_y=False)
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
    elif dataset == 'boston':
        data = fetch_openml(data_id=853, return_X_y=False)
        X = data.data
        y = data.target
        y = (y=='P').astype('int')#.values
        feature_names = data.feature_names
        target_names = data.target_names
    
    elif dataset == 'credit-card': # Warning: 1) Huge dataset. 2) Etremely Imbalanced: Accuracy not appropriate, use Recall/F1
        data = fetch_openml(data_id=1597, return_X_y=False)
        X = data.data
        y = data.target.astype('int')#.values
        feature_names = data.feature_names
        target_names = data.target_names
   
    elif dataset == 'churn': # Warning: Imbalanced! Accuracy not appropriate, use Recall/F1
        data = fetch_openml(data_id=40701, return_X_y=False)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        #del df['area_code'] #drop categorical attribute
        cat_names = ['area_code']#, 'education']#, 'marital-status', 'occupation', 'relationship', 'native-country']
        continuous_names = [x for x in df.columns if x not in cat_names]
        X = df.values
        feature_names = df.columns
        y = data.target.astype('int')#.values
        target_names = data.target_names

        idx = np.random.choice(df.index, 5000)
        X, y = X[idx, :], y[idx]
    
    elif dataset == 'news': # Warning: Imbalanced! Accuracy not appropriate, use Recall/F1
        data = fetch_openml(data_id=4545, return_X_y=False)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        X = df.values
        feature_names = df.columns
        y = data.target.astype('int')#.values
        y = (y>y.mean()).astype('int')#.values
        target_names = data.target_names
        
        #idx = np.random.choice(df.index, 1000)
        #X, y = X[idx, :], y[idx]
        
    elif dataset == 'boson': # Warning: Imbalanced! Accuracy not appropriate, use Recall/F1
        data = fetch_openml(data_id=23512, return_X_y=False)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        X = df.values
        feature_names = df.columns
        y = data.target.astype('int')#.values
        target_names = data.target_names
    
    
    elif dataset == 'adult-num': # Warning: Imbalanced! Accuracy not appropriate, use Recall/F1
        data = fetch_openml(data_id=1590, return_X_y=False)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        to_keep  = ['capital-gain', 'age', 'capital-loss', 'education-num', 'hours-per-week']
        #cat_names = ['sex']
        continuous_names = list(set(to_keep) - set(cat_names))
        df = df[to_keep]
        df = pd.get_dummies(df)
        cat_names = list(set(df.columns) - set(continuous_names))
        X = df.values
        feature_names = df.columns
        y = (data.target=='>50K').astype('int')#.values
        target_names = data.target_names
        
        print('taking only 5000 instances')
        idx = np.random.choice(df.index, 5000, replace=False)
        X, y = X[idx, :], y[idx]
        

    elif dataset == 'adult-cat':
        data = fetch_openml(data_id=1590, return_X_y=False)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        to_keep  = ['capital-gain', 'age', 'capital-loss', 'education-num', 'hours-per-week', 'sex', 'race', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
        cat_names = ['sex', 'race', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
        continuous_names = [x for x in to_keep if x not in cat_names]
        df = df[to_keep]
        df = pd.get_dummies(df)
        cat_names = [x for x in list(df.columns) if x not in continuous_names]
        X = df.values
        feature_names = df.columns
        y = (data.target=='>50K').astype('int')#.values
        target_names = data.target_names
        
        print('taking only 10000 instances')
        idx = np.random.choice(df.index, 10000, replace=False)
        X, y = X[idx, :], y[idx]
        
    elif dataset == 'german':
        data = fetch_openml(data_id=31, return_X_y=False)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        to_keep  = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 'num_dependents', 'credit_history'] #, 'checking_status', 'purpose', 'saving_status', 'employment', 'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']
        cat_names = ['credit_history' ]
        continuous_names = [x for x in to_keep if x not in cat_names]
        df = df[to_keep]
        df = pd.get_dummies(df)
        cat_names = [x for x in list(df.columns) if x not in continuous_names]
        X = df.values
        feature_names = df.columns
        y = (data.target=='good').astype('int')#.values
        target_names = data.target_names
        
        #print('taking only 1000 instances')
        #idx = np.random.choice(df.index, 10000, replace=False)
        #X, y = X[idx, :], y[idx]

    else:
        raise ValueError


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    if standardize:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    feature_names = list(feature_names)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    X_train = pd.DataFrame(X_train, index=y_train.index, columns=feature_names)
    X_test = pd.DataFrame(X_test, index=y_test.index, columns=feature_names)

    return X_train, X_test, y_train, y_test, scaler, feature_names, target_names, cat_names

