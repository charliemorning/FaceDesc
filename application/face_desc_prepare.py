# -*- coding: utf-8 -*-


import os
import pickle

import cv2
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from scipy import misc
import datetime

from util.face_repr import get_face_repr
from application.face_desc import predict_gender

def prepare_face_classification(path):
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

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf.fit(X, Y)

    f = open('../models/face_classification/svm.model', 'wb')
    pickle.dump(clf, f)
    f.close()


def save_face(saveDir,face_area):
    try:
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        time_str = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")
        face_file_name = time_str + ".jpg"
        save_path = os.path.join(saveDir, face_file_name)
        misc.imsave(save_path,face_area)
        return save_path
    except Exception as e:
        raise e

def validate_data(path):
    for label in os.listdir(path):
        for image_file_name in os.listdir(path + label):
            image = misc.imread(path + label + "/" + image_file_name)

            print(label, image_file_name, predict_gender(image))

if __name__ == "__main__":
    # prepare_face_classification("C:/developer/project/pycharm-workspace/facedesc/data/image/face_recognition/")
    # validate_data("C:/developer/project/pycharm-workspace/facedesc/data/image/face_recognition/")
    validate_data("C:/Users/Charlie/Desktop/2/")