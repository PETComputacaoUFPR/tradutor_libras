# Collects images from user to create datasets
import os
import cv2

DATASET_SIZE = 10  # how many images will be collected
name = input("Your name: ")  # your name (to differentiate images)


# get symbols from "symbols" file
with open("symbols", "r") as file:
    lines = file.read().splitlines()
    symbols = lines[0].strip()

# creates images path
DATA_DIR = './imagens'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# for some reason, the code prints an error the first time it runs the frame
# that's literally the only usage of this code block
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow('frame', frame)

# loops through symbols
for symbol in symbols:
    if not os.path.exists(os.path.join(DATA_DIR, symbol)):
        os.makedirs(os.path.join(DATA_DIR, symbol))

    # path of symbol
    symbol_path = os.path.join(DATA_DIR, symbol)

    # only collects images if there are less images than DATASET_SIZE
    if len(symbol_path) >= DATASET_SIZE:
        continue

    # collects remaining images
    print('Collecting data for class {}'.format(symbol))
    counter = len(os.listdir(symbol_path)) + 1
    while counter <= dataset_size:
        ret, frame = cap.read()
        # the output for pressing ENTER is 13
        if cv2.waitKey(1) == 13:
            print('image {} collected'.format(counter))
            cv2.imwrite(os.path.join(DATA_DIR, symbol, '{}_{}.jpg'.format(name, counter)), frame)
            counter += 1
        cv2.imshow('frame', frame)

# ends application after capturing all images
print("Images successfully captured!")
cap.release()
cv2.destroyAllWindows()
