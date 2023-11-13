# unites datasets of different people into one

import os
import pickle

# final dataset
data = {"features": [], "labels": []}

# directory of this file
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(WORKING_DIR, "data/")

# loops through datasets
for dataset in os.listdir(DATA_DIR):
    # only merges pickles that start with "partial"
    if ("partial" not in dataset or "pickle" not in dataset):
       continue
    with open(os.path.join(DATA_DIR, dataset), "rb") as new_data:
        data_aux = pickle.load(new_data)
        data["features"].extend(data_aux["features"])
        data["labels"].extend(data_aux["labels"])
    print(f"{dataset} merged")

# saves as new dataset
dataset_path = os.path.join(WORKING_DIR, "base_dataset.pickle")
with open(dataset_path, "wb") as dataset:
    pickle.dump(data, dataset)

print("Dataset merged sucessfuly!")
