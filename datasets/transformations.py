# Functions that take a pickle file and makes transformations such that
# can create a new dataset
import os
import pickle

# substracts each coordinate by the smallest value (ignores z values)
def minimum (features):
    # get smallest x and y
    min_x, min_y = features[0], features[1]
    for i, value in enumerate(features[0:-1]):
        # x value
        if i % 3 == 0:
            min_x = min(min_x, value)
        # y value
        elif i % 3 == 1:
            min_y = min(min_y, value)

    # new features
    new_features = []
    for i, value in enumerate(features):
        # x value
        if i % 3 == 0:
            new_features.append(value - min_x)
        # y value
        elif i % 3 == 1:
            new_features.append(value - min_y)
    # adds back the orientation of hand (left/right)
    new_features.append(features[-1])
    return new_features
    
