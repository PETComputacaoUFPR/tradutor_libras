# Functions that take a pickle file and makes transformations such that
# can create a new dataset
import os
import pickle
from math import sqrt

# substracts each coordinate by the smallest value
def minimum (features):
    # get smallest x and y
    min_x, min_y, min_z = features[0], features[1], features[2]
    for i, value in enumerate(features[0:-1]):
        # x value
        if i % 3 == 0:
            min_x = min(min_x, value)
        # y value
        elif i % 3 == 1:
            min_y = min(min_y, value)
        # z value
        else:
            min_z = min(min_z, value)

    # new features
    new_features = []
    for i, value in enumerate(features):
        # x value
        if i % 3 == 0:
            new_features.append(value - min_x)
        # y value
        elif i % 3 == 1:
            new_features.append(value - min_y)
        # z value
            new_features.append(value - min_z)
    # adds back the orientation of hand (left/right)
    new_features.append(features[-1])
    return new_features


# uses 3D coordinate system with origin in point 0
def geometric (features):
    # gets values from origin (point 0)
    x_0, y_0, z_0 =  features[0], features[1], features[2]

    # loops through other points (last one is left/right hand)
    new_features = []
    for i, value in enumerate(features[3:-1]):
        value = value
        if i % 3 == 0:
            new_features.append(value - x_0)
        elif i % 3 == 1:
            new_features.append(value - y_0)
    new_features.append(features[-1])
    return new_features
    

# saves coordinates as unitary vectors (from point 0) and their distances
# uses 3D space
def vectorial3D (features):
    # gets values from origin (point 0)
    x_0, y_0, z_0 = features[0], features[1], features[2]

    new_features = []
    # loops through other points (last one is left/right hand)
    for i, value in enumerate(features[3:-1:3]):
        x, y, z = features[i], features[i+1], features[i+2]
        distance = sqrt((x - x_0)**2 + (y - y_0)**2 + (z - z_0)**2)
        new_features.append(x - x_0)
        new_features.append(y - y_0)
        new_features.append(z - z_0)
        new_features.append(distance)
    new_features.append(features[-1])
    return new_features
