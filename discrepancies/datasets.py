import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_wine, fetch_20newsgroups_vectorized, fetch_kddcup99, make_moons


RANDOM_STATE = 42

def get_dataset(dataset='half-moons',
                n_samples=100,
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

    if dataset == 'breast-cancer':
        data = load_breast_cancer(return_X_y=False)
        X = data.data
        y = data.target.reshape(len(data.target),-1)

    elif dataset == 'load-wine':
        data = load_wine_cancer(return_X_y=False)
        X = data.data
        y = data.target.reshape(len(data.target),-1)

    elif dataset == 'half-moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=RANDOM_STATE)

    elif dataset == '20-newsgroups':
        data = fetch_20newsgroups_vectorized(subset='all', return_X_y=False)
        X = data.data
        y = data.target.reshape(len(data.target),-1)

    elif dataset == 'kddcup99':
        data = fetch_kddcup99(subset=None, return_X_y=False)
        X = data.data
        y = data.target.reshape(len(data.target),-1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    if standardize:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    return X_train, X_test, y_train, y_test, scaler

