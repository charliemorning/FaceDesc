# -*- coding: utf-8 -*-

import cv2
import keras
import numpy as np

from network.mobilenet import DepthwiseConv2D, relu6

print("load age model...")

birth_model = keras.models.load_model("../models/birth.mobilenet.07-0.81.model",
                                      custom_objects={
                                'relu6': relu6,
                                'DepthwiseConv2D': DepthwiseConv2D
                                })

BIRTH_YEAR = [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]

def predict_birth(image):
    image = cv2.resize(image, (224, 224))
    image = np.asarray(image, dtype="float32") / 255.0
    birth = birth_model.predict(np.array([image]))
    index = np.argmax(birth)
    # print(BIRTH_YEAR[index])

    return BIRTH_YEAR[index]
    # return datetime.datetime.now().year - int(BIRTH_YEAR[index]) - np.random.randint(-3, 3)