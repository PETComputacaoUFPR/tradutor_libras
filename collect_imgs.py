import os
import cv2

dataset_size = 10  # how many images will be collected
name = input("Your name: ")


# get symbols from "symbols" file
with open("symbols", "r") as file:
    lines = file.read().splitlines()
    symbols = lines[0].strip()

DATA_DIR = './data'
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

    print('Collecting data for class {}'.format(symbol))

    # only collect remaining images
    counter = len(os.listdir(os.path.join(DATA_DIR, symbol))) + 1
    while counter <= dataset_size:
        ret, frame = cap.read()
        # the output for pressing ENTER is 13
        if cv2.waitKey(1) == 13:
            print('image {} collected'.format(counter))
            cv2.imwrite(os.path.join(DATA_DIR, symbol, '{}_{}.jpg'.format(name, counter)), frame)
            counter += 1
        cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
