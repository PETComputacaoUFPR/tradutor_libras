import pickle

import cv2
import mediapipe as mp
import numpy as np

# controls min probability to classify class
MIN_PROB = 0.0

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

while True:

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        data_aux = []
        x_0 = hand_landmarks.landmark[0].x
        y_0 = hand_landmarks.landmark[0].y
        x_min, x_max = x_0, x_0
        y_min, y_max = y_0, y_0
        for i in range(1, len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - x_0)
            data_aux.append(y - y_0)

            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        x1 = int(x_min * W) - 10
        y1 = int(y_min * H) - 10

        x2 = int(x_max * W) - 10
        y2 = int(y_max * H) - 10

        #print(model.predict_proba([np.asarray(data_aux)])[0])
        if max(model.predict_proba([np.asarray(data_aux)])[0]) < MIN_PROB:
            predicted_character = ''
        else:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
