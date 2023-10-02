# Trains the classifier (RandomForest)
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Loads dataset
data = pickle.load(open('../datasets/base_dataset.pickle', 'rb'))

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(data["features"], 
    data["labels"], test_size=0.2, shuffle=True, stratify=data["labels"])

# Trains Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Tests Model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Close Dataset
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
