# trains all classifiers
# classifiers are choosen in below variables
# transformations in dataset
import sys
import os

# directory of this file
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
def path_to(p):
    return os.path.join(WORKING_DIR, p)

sys.path.append(path_to("../datasets"))
from transformations import minimum, geometric, vectorial2D

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 

# other modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pickle
import joblib

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
def weighted_accuracy(y_true, y_pred):
    recall_array = recall_score(y_true, y_pred, average=None)
    weights_total = 0
    result = 0
    for recall, weight in zip(recall_array, LETTERS_FREQUENCY):
        weights_total += weight
        result += recall * weight
    return result / weights_total
weighted_accuracy_score = make_scorer(weighted_accuracy)


# functions that train the models
# returns the model trained and its accuracy in cross val score
# if feature scaling is necessary, makes minmax
def train_model (model, data, scaling=False):
    if scaling:
        scaler = MinMaxScaler()
        features = scaler.fit_transform(data["features"])
    else:
        features = data["features"]
    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(features,
        data["labels"], test_size=0.2, shuffle=True, stratify=data["labels"])

    # train model
    model.fit(x_train, y_train)

    # test model
    y_predict = model.predict(x_test)
    score = weighted_accuracy(y_test, y_predict)

    return model, score

# get base_dataset
data_path = path_to("../datasets/base_dataset.pickle")
data = pickle.load(open(data_path, "rb"))

# list of models and their names (names only to make prints easier)
models = [
    {"name": "RandomForest", "classifier": RandomForestClassifier(n_jobs=-1), "scale": False},
    {"name": "LogisticRegression", "classifier": SGDClassifier(loss="log_loss"), "scale": True},
    {"name": "SVM", "classifier": SVC(), "scale": True},
    {"name": "KNN", "classifier": KNeighborsClassifier(), "scale": True}
]

# list of changed_models and their names (names only to make prints easier)
changed_datasets = [
    {"name": "Minimum", "transformation": minimum, "dataset": {}},
    {"name": "Geometric", "transformation": geometric, "dataset": {}},
    {"name": "Vectorial2D", "transformation": vectorial2D, "dataset": {}}
]
for option in changed_datasets:
    new_features = []
    for observation in data["features"]:
        new_features.append(option["transformation"](observation))
    option["dataset"]["features"] = new_features
    option["dataset"]["labels"] = data["labels"]
    option["dataset"]["person_id"] = data["person_id"]

# trains all models and prints their results
for model in models:
    for dataset in changed_datasets:
        _, accuracy = train_model(model["classifier"], dataset["dataset"], scaling=model["scale"])
        print(model["name"], dataset["name"], accuracy)
