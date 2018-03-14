# -*- coding: utf-8 -*-

import os
import sys

import cv2 as cv
import keras
import numpy as np

from network.mobilenet import DepthwiseConv2D, relu6

print("load gender model...")

gender_model = keras.models.load_model("../models/gender/gender.mobilenet.augment.18-0.00.model",
                                custom_objects={
                                'relu6': relu6,
                                'DepthwiseConv2D': DepthwiseConv2D
                                })

def predict_gender(image):
    image = cv.resize(image, (224, 224))
    image = np.asarray(image, dtype="float32") / 255.0
    gender = gender_model.predict(np.array([image]))
    print("gender", gender)

    if gender[0][0] > gender[0][1]:

        return "female"
    else:
        return "male"

if __name__ == "__main__":

    base_path = sys.argv[1]
    img_ext = "png"

    for image_filename in os.listdir(base_path):
        if image_filename.endswith(img_ext):
            try:
                image = cv.imread(base_path + image_filename)
                print(image_filename, predict_gender(image))
            except:
                print("error", image_filename)