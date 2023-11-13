# Creates dataset with hand coordinates positions
# Is used as a base dataset for other datasets
import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# directory of this file
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
def path_to(p):
    return os.path.join(WORKING_DIR, p)

# name of file
FILE_NAME = input("file name (without .pickle): ")
FILE_NAME += ".pickle"

# get symbols from "symbols" file
symbols_path = path_to("../symbols")
with open(symbols_path, "r") as file:
    lines = file.read().splitlines()
    symbols = lines[0].strip()

# Used to get hand's coordinates
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# change confidence detection if necessary
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# path of images
IMAGES_DIR = path_to("../images")

# destiny directory
DATA_DIR = path_to("data")

# dictionary that will save data for dataset
data = {"features": [], "labels": []}

# loops through directories (each named with a label)
total_counter = 0
total_size = 0
for dir_ in symbols:
    counter = 0
    symbol_size = len(os.listdir(os.path.join(IMAGES_DIR, dir_)));
    # loops through images
    for img_path in os.listdir(os.path.join(IMAGES_DIR, dir_)):
        data_aux = []

        # reads image and converts it to RGB
        img = cv2.imread(os.path.join(IMAGES_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # only adds data if there is only one hand detected
        if results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) == 1:
            counter += 1
            hand_landmarks = results.multi_hand_landmarks[0]
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                data_aux.append(x)
                data_aux.append(y)
                data_aux.append(z)
            # if hand is left or right
            is_left = (results.multi_handedness[0].classification[0].label == "Left")
            data_aux.append(int(is_left))
            data["features"].append(data_aux)
            data["labels"].append(dir_)
    print(f"{dir_}: {counter}/{symbol_size}")
    total_counter += counter
    total_size += symbol_size

# Creates dataset
with open(os.path.join(DATA_DIR, FILE_NAME), "wb") as dataset:
    pickle.dump(data, dataset)

print("Dataset successfully created!")
print(f"Images saved: {total_counter}/{total_size}")
