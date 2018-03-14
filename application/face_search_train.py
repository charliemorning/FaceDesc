# -*- coding: utf-8 -*-

import os
import pickle

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors




def train_knn(x, y):

    neigh = NearestNeighbors(5, 1.0, metric="euclidean")
    neigh.fit(x, y)
    return neigh


def save_model(path, model):
    f = open(path, 'wb')
    pickle.dump(model, f)
    f.close()


def load_data_from_image(path):
    from util.face_repr import get_face_repr
    X = []
    Y = []
    for label in os.listdir(path):
        for image_file_name in os.listdir(path + label):
            image = cv2.imread(path + label + "/" + image_file_name)
            vect = get_face_repr(image)
            X.append(vect[0])
            Y.append(label)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def load_data_from_text(path):
    X = []
    Y = []
    with open(path, "r") as f:
        for line in f:
            vect_str, label = line.strip().split(",")
            vect = [float(v) for v in vect_str.split(" ")]
            X.append(vect)
            Y.append(label)

        f.close()

    X = np.array(X)
    Y = np.array(Y)

    return X, Y



if __name__ == '__main__':
    X, Y = load_data_from_text("../data/face_recognition_face_rep.txt")
    model = train_knn(X, Y)
    save_model('../models/face_classification/knn.model', model)

