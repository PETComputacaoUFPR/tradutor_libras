# gambiarra para poder unir arquivos pickle de diferentes pessoas

import os
import pickle

# final dataset
data = {"features": [], "labels": []}

DATA_DIR = "./data"

# loops through datasets
for dataset in os.listdir(DATA_DIR):
    # only merges pickles that start with "partial"
    if ("partial" not in dataset or "pickle" not in dataset):
       continue
    with open(os.path.join(DATA_DIR, dataset), "rb") as new_data:
        data.update(pickle.load(new_data))
    print(f"{dataset} merged")

# saves as new dataset
with open("base_dataset.pickle", "wb") as dataset:
    pickle.dump(data, dataset)

print("Dataset merged sucessfuly!")
