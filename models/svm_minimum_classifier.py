# Trains the classifier (SVM Classifier)
import pickle
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import sys
import os

WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
def path_to(p):
    return os.path.join(WORKING_DIR, p)

sys.path.append(path_to("../datasets"))
from transformations import minimum

# Loads dataset
data = pickle.load(open(path_to('../datasets/base_dataset.pickle'), 'rb'))


# Transforms dataset
new_features = []
for observation in data["features"]:
    new_features.append(minimum(observation))

# MinMaxScaling
scaler = MinMaxScaler()
new_features = scaler.fit_transform(new_features)

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(new_features,
    data["labels"], test_size=0.2, shuffle=True, stratify=data["labels"])

# Trains Model
model = SVC()
model.fit(x_train, y_train)

# Tests Model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Save Model and Scaler
f = open('smv_minimum.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
joblib.dump(scaler, "scaler_minimum.joblib")
