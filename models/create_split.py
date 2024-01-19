# Splits dataset into train and test and stores in TrainTestData folder (creates it if necessary)

import pickle

from sklearn.model_selection import train_test_split
import numpy as np

import sys
import os

# gets path of current directory (where this file exists)
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
def path_to(p):
    return os.path.join(WORKING_DIR, p)

# Loads dataset
data = pickle.load(open(path_to('../datasets/base_dataset.pickle'), 'rb'))

X = data["features"]
y = data["labels"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=data["labels"])

# Creates folder (if necessary)
if not os.path.exists(path_to("TrainTestData")):
    os.makedirs(path_to("TrainTestData"))

# Creates files for data
data_train = {"features": X_train, "labels": y_train}
data_test = {"features": X_test, "labels": y_test}

with open(path_to("TrainTestData/train_data.pickle"), "wb") as dataset:
    pickle.dump(data_train, dataset)
with open(path_to("TrainTestData/test_data.pickle"), "wb") as dataset:
    pickle.dump(data_test, dataset)