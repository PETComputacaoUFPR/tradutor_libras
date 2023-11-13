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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# other modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import joblib


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
    score = accuracy_score(y_predict, y_test)

    return model, score

# get base_dataset
data_path = path_to("../datasets/base_dataset.pickle")
data = pickle.load(open(data_path, "rb"))

# list of models and their names (names only to make prints easier)
models = [
    {"name": "RandomForest", "classifier": RandomForestClassifier(), "scale": False},
    {"name": "LogisticRegression", "classifier": LogisticRegression(), "scale": True},
    {"name": "SVM", "classifier": SVC(), "scale": True}
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

# trains all models and prints their results
for model in models:
    for dataset in changed_datasets:
        _, accuracy = train_model(model["classifier"], dataset["dataset"], scaling=model["scale"])
        print(model["name"], dataset["name"], accuracy)
