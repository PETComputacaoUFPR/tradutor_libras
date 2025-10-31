# Collects images from user to create datasets
import os
import cv2
import mediapipe as mp

DATASET_SIZE = 20  # how many images will be collected
ENTER_KEY = 13

# directory of this file
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# get symbols from "symbols" file
symbols_path = os.path.join(WORKING_DIR, "symbols")
with open(symbols_path, "r") as file:
    lines = file.read().splitlines()
    symbols = lines[0].strip()

# creates images path
DATA_DIR = os.path.join(WORKING_DIR, "images")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# for some reason, the code prints an error the first time it runs the frame
# that's literally the only usage of this code block
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow('frame', frame)

# 0 -> symbol; 1 -> counter
image_info = ["", 0] 

# allows for saving images via mouse and enter
def save_image(params):
    print(f"image {params[1]}/{DATASET_SIZE} collected")
    cv2.imwrite(os.path.join(DATA_DIR, params[0], '_{}.jpg'.format(params[1])), frame)
    params[1] += 1

def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        save_image(image_info)

cv2.setMouseCallback("frame", mouse_callback)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# loops through symbols
for symbol in symbols:
    image_info[0] = symbol
    if not os.path.exists(os.path.join(DATA_DIR, symbol)):
        os.makedirs(os.path.join(DATA_DIR, symbol))

    # path of symbol
    symbol_path = os.path.join(DATA_DIR, symbol)

    # only collects images if there are less images than DATASET_SIZE
    if len(os.listdir(symbol_path)) >= DATASET_SIZE:
        continue

    # collects remaining images
    print('Collecting data for class {}'.format(symbol))
    image_info[1] = len(os.listdir(symbol_path)) + 1
    while image_info[1] <= DATASET_SIZE:
        ret, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # if there's a hand, tries to detect symbol
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
        
            # iterate over each detected hand
            for hand_landmarks in results.multi_hand_landmarks:

                all_x = []
                all_y = []

                # iterate for all 21 landmarks
                for lm in hand_landmarks.landmark:
                    h, w, c = frame.shape
                    center_x = int(lm.x * w)
                    center_y = int(lm.y * h)
                    all_x.append(center_x)
                    all_y.append(center_y)

                # coordinates and padding
                x1 = min(all_x) - 20
                y1 = min(all_y) - 20
                x2 = max(all_x) + 20
                y2 = max(all_y) + 20

                # draw a bow around the hands
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        if cv2.waitKey(1) == ENTER_KEY:
            save_image(image_info)

        cv2.imshow('frame', frame)

# ends application after capturing all images
print("Images successfully captured!")
cap.release()
cv2.destroyAllWindows()