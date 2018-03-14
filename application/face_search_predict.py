# -*- coding: utf-8 -*-

import pickle

from application.face_search_train import load_data_from_text


def load_model(path):
    model_file = open(path, 'rb')
    model = pickle.load(model_file)
    return model

model = load_model('../models/face_classification/knn.model')

X, Y = load_data_from_text("../data/face_recognition_face_rep.txt")

def search_face(vect):

    result = model.kneighbors(vect, n_neighbors=5, return_distance=True)

    for i in range(result[0].__len__()):
        yield (result[0][i], Y[result[1][i]])
