# Application of model
import pickle
import joblib
import sys

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(1, "./datasets")
from transformations import minimum

# controls min probability to classify class
MIN_PROB = 0.0

# Loads model
model_dict = pickle.load(open('./models/svm_minimum.p', 'rb'))
model = model_dict['model']

# Creates panel and hand application
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Only one hand will be detected per frame
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

while True:
    # Creates frame and changed color system to RGB
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if there's a hand, tries to detect symbol
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draws hand's coordinates
        mp_drawing.draw_landmarks(
            frame,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

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
        input_values = minimum(data_aux)
        scaler = joblib.load("models/scaler.joblib")
        input_values = scaler.transform(np.reshape(input_values, (1,-1)))[0]
        #if max(model.predict_proba([np.asarray(input_values)])[0]) < MIN_PROB:
         #   predicted_character = ''
        #else:
        prediction = model.predict([np.asarray(input_values)])
        predicted_character = prediction[0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


# Destroys panel
cap.release()
cv2.destroyAllWindows()
