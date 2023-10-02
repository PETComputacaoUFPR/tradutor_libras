# Creates dataset with hand coordinates positions
# Is used as a base dataset for other datasets
import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# get symbols from "symbols" file
with open("../symbols", "r") as file:
    lines = file.read().splitlines()
    symbols = lines[0].strip()

# Used to get hand's coordinates
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# change confidence detection if necessary
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# path of images
DATA_DIR = '../images'

data = []   # features (input of model)
labels = []  # symbols (output of model)

# loops through directories (each named with a label)
for dir_ in symbols:
    # loops through images
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        # reads image and converts it to RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # only adds data if there is only one hand detected
        if len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                data_aux.append(x)
                data_aux.append(y)
                data_aux.append(z)

            data.append(data_aux)
            labels.append(dir_)

# Creates dataset
f = open('base_dataset.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print("Dataset successfully created!")
