# -*- coding: utf-8 -*-


import tensorflow as tf

import util.facenet as facenet

import cv2
import numpy as np

# model = "../models/facenet/20170511-185253/20170511-185253.pb"
model = "../models/facenet/20170512-110547/20170512-110547.pb"
global_sess = None

with tf.Graph().as_default():
    with tf.Session() as sess:

        print("Load facenet model...")
        # Load the model
        facenet.load_model(model)
        global_sess = sess
        print("Facenet model loaded done...")


def get_face_repr(image):
    global global_sess

    image = cv2.resize(image, (160, 160))
    prewhitened = facenet.prewhiten(image)
    images = np.asarray([prewhitened])

    images_placeholder = global_sess.graph.get_tensor_by_name("input:0")
    embeddings = global_sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = global_sess.graph.get_tensor_by_name("phase_train:0")
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    vects = sess.run(embeddings, feed_dict=feed_dict)
    return vects

