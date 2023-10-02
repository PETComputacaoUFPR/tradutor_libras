# Creates dataset with hand coordinates positions based on minimum coordinates
import pickle
from transformations import minimum

with open("base_dataset.pickle", "rb") as dataset:
    old_dataset = pickle.load(dataset)

with open("minimum_dataset.pickle", "wb") as dataset:
    new_data = {"features": [], "labels": old_dataset["labels"]}
    for i in range(len(old_dataset["features"])):
        new_data["features"].append(minimum(old_dataset["features"][i]))
    pickle.dump(new_data, dataset)

print("Dataset successfully created!")
