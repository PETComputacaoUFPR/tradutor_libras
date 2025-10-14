# Application of model
import pickle
import joblib
import sys
import os

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(1, "./models")
from models.LibrasModel import LibrasModel

def closeApp(exitCode):
    cap.release()
    cv2.destroyAllWindows()
    exit(exitCode)


# how many hands to detect (in max)
MAX_HANDS = 2

# controls min probability to classify class
MIN_PROB = 0.0

# Exit key code
QUIT_KEY = 113

# Window name
WINDOW_NAME = "App"

# directory of this file
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# Loads model
model_path = os.path.join(WORKING_DIR, "models/best_model.sav")
base_model = pickle.load(open(model_path, 'rb'))
model = LibrasModel(base_model, has_z=False)


# predicts symbol from one hand
# ialso returns the box limits of hand
def predict(hand_landmarks):
        # Gets values for prediction and to draw box around hand
        data_aux = []
        x_min, x_max = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].x
        y_min, y_max = hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].y
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            z = hand_landmarks.landmark[i].z
            data_aux.append(x)
            data_aux.append(y)
            data_aux.append(z)

            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        x1 = int(x_min * W) - 10
        y1 = int(y_min * H) - 10

        x2 = int(x_max * W) - 10
        y2 = int(y_max * H) - 10
        data_aux.append(results.multi_handedness[0].classification[0].label == "Left")

        # prediction
        x = np.array([data_aux])
        predicted_character = model.predict(x)[0]
        return predicted_character, (x1, y1), (x2, y2) 


# Initi video cpture and verify camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("\033[31mNão foi possível abrir a camera, saindo do programa !!!\033[0m")
    closeApp(1)
else:
    print("Camera inicada")

# Create cv2 window and close callback
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)

# Creates panel and hand application
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=MAX_HANDS, min_detection_confidence=0.3)

while cv2.waitKeyEx(1) != QUIT_KEY:
    # Creates frame and changed color system to RGB
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if there's a hand, tries to detect symbol
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
              
        for hand_landmarks in results.multi_hand_landmarks:
            # Draws hand's coordinates
            #mp_drawing.draw_landmarks(
            #    frame,  # image to draw
            #    hand_landmarks,  # model output
            #    mp_hands.HAND_CONNECTIONS,  # hand connections
            #    mp_drawing_styles.get_default_hand_landmarks_style(),
            #    mp_drawing_styles.get_default_hand_connections_style())

            predicted_character, p1, p2 = predict(hand_landmarks)
            x1, y1 = p1
            x2, y2 = p2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # Close wwindow when "x" clicked
    if not cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE):
        break
    cv2.imshow(WINDOW_NAME, frame)


# Destroys panel
closeApp(0)
