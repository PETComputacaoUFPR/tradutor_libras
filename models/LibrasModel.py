"""
    Blueprint for all models
    Also adds useful metrics
"""

import time
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score, make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator

# function that calculates weighted_accuracy
# weights are basead on the frequency of the letters in the portuguese alphabet 
# source: https://pt.wikipedia.org/wiki/Alfabeto_portugu%C3%AAs#Frequ%C3%AAncia_da_ocorr%C3%AAncia_de_letras
# H, K, J, X and Z are not present
LETTERS_FREQUENCY = [
    14.63,
    1.04,
    3.88,
    5.01,
    12.57,
    1.02,
    1.30,
    6.18,
    2.78,
    4.74,
    5.05,
    10.73,
    2.52,
    1.20,
    6.53,
    7.81,
    4.34,
    4.63,
    1.67,
    0.01,
    0.01,
]
def weighted_accuracy_score(y_true, y_pred):
    recall_array = recall_score(y_true, y_pred, average=None)
    weights_total = 0
    result = 0
    for recall, weight in zip(recall_array, LETTERS_FREQUENCY):
        weights_total += weight
        result += recall * weight
    return result / weights_total
weighted_accuracy_scorer = make_scorer(weighted_accuracy_score)

class LibrasModel(BaseEstimator):
    def __init__(self, model, has_z=True):
        self.model = model
        self.has_z = has_z


    def fit(self, X, y):
        X = self.transform_data(X)
        self.model.fit(X, y)


    def predict(self, X):
        X = self.transform_data(X)
        y_pred = self.model.predict(X)
        return y_pred
    

    def cross_val(self, X, y, scoring, cv=5, mean=False):
        X = self.transform_data(X)
        scores = cross_val_score(self.model, X, y, cv=cv, n_jobs=-1, scoring=scoring)
        if mean:
            return np.mean(scores)
        return scores
    

    def transform_data(self, X):
        X_copy = np.copy(X)
        X_copy = self.centralize(X_copy)
        X_copy = self.fix_hand(X_copy)
        if not self.has_z:
            X_copy = self.to_2d(X_copy)
        return X_copy


    # X: each row has 64 elements [x0, y0, z0, ..., x20, y20, z20, left_hand?]
    def centralize(self, X):
        X[:, 0:-1:3] -= X[:, 0][:, np.newaxis]
        X[:, 1:-1:3] -= X[:, 1][:, np.newaxis]
        X[:, 2:-1:3] -= X[:, 2][:, np.newaxis]
        X = X[:, 3:]
        return X
    

    def fix_hand(self, X):
        X[:, 0::3] = (-2 * X[:, -1][:, np.newaxis] + 1) * X[:, 0::3]
        X = X[:, :-1]
        return X
    

    def to_2d(self, X):
        d = np.zeros(X.shape[1], dtype=bool)
        d[2::3] = True
        X = np.delete(X, d, axis=1)
        return X
