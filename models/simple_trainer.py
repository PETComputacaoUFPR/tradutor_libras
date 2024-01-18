# Trains the classifier (RandomForest)
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np

import sys
import os

WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
def path_to(p):
    return os.path.join(WORKING_DIR, p)

sys.path.append(path_to("../datasets"))
from transformations import geometric

# Loads dataset
data = pickle.load(open(path_to('../datasets/base_dataset.pickle'), 'rb'))


# Transforms dataset
new_features = []
for observation in data["features"]:
    new_features.append(geometric(observation))

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(new_features,
    data["labels"], test_size=0.2, shuffle=True, stratify=data["labels"])

# Trains Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Tests Model
y_predict = model.predict(x_test)
score = balanced_accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Save Model
f = open('geometric_model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
