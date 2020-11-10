import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons

from discrepancies.pool import autogluon_pool

data = load_breast_cancer(return_X_y=False)

X = data.data
y = data.target.reshape(len(data.target),-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

pool = autogluon_pool().fit(X_train, y_train, time_limit=2)

performance, leaderboard = pool.get_leaderboard(X_test, y_test)
print(performance)
print(leaderboard)

print(pool.individual_model_predictions(X_test))

print('stop')