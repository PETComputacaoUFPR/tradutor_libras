"""
    Creates the best model, according to the analysis in the jupyter notebooks
    To see info about best model, see conclusion in "best_model.ipynb"

    Best Model: SVM with geometric2D transformation, kernel = "rbf", C = 40 and gamma = 5
    obs: by using 2D instead of 3D, size of model drops from 1.2M to 816K
"""

# importing important libraries
# transformations library
from transformations import geometric2D
# models
from sklearn.svm import SVC
# loading data
import pickle
# other modules
import numpy as np
import sys
import os

# gets path of current directory (where this file exists)
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
def path_to(p):
    return os.path.join(WORKING_DIR, p)

# get dataset
data_path = path_to("TrainTestData/train_data.pickle")
data = pickle.load(open(data_path, "rb"))

# Geometric transformation
geometric2D_X = []
for observation in data["features"]:
    geometric2D_X.append(geometric2D(observation))

svm = SVC(kernel="rbf", C=40, gamma=5)
svm.fit(geometric2D_X, data["labels"])

pickle.dump(svm, open(path_to("best_model.sav"), "wb"))