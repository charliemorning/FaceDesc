# -*- coding: utf-8 -*-


import os
import csv
import numpy as np
import math
from skimage import io, transform
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import cv2


def load_fer2013(path):
    pass


def get_keras_image_data_generator_from_directory(
        path,
        target_size,
        fit_x_train=None,
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix=None,
        save_format="png",
        follow_links=False,
        interpolation="nearest",
        # parameters of ImageDataGenerator
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=K.image_data_format()):
    """"""

    generator = ImageDataGenerator(featurewise_center=featurewise_center,
                                   samplewise_center=samplewise_center,
                                   featurewise_std_normalization=featurewise_std_normalization,
                                   samplewise_std_normalization=samplewise_std_normalization,
                                   zca_whitening=zca_whitening,
                                   zca_epsilon=zca_epsilon,
                                   rotation_range=rotation_range,
                                   width_shift_range=width_shift_range,
                                   height_shift_range=height_shift_range,
                                   shear_range=shear_range,
                                   zoom_range=zoom_range,
                                   channel_shift_range=channel_shift_range,
                                   fill_mode=fill_mode,
                                   cval=cval,
                                   horizontal_flip=horizontal_flip,
                                   vertical_flip=vertical_flip,
                                   rescale=rescale,
                                   preprocessing_function=preprocessing_function,
                                   data_format=data_format)
    if fit_x_train is not None:
        generator.fit(fit_x_train)
    return generator.flow_from_directory(path,
                                  target_size=target_size,
                                  color_mode=color_mode,
                                  classes=classes,
                                  class_mode=class_mode,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  seed=seed,
                                  save_to_dir=save_to_dir,
                                  save_prefix=save_prefix,
                                  save_format=save_format,
                                  follow_links=follow_links,
                                  interpolation=interpolation)


def load_labels(path):
    labels = []
    with open(path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            labels.append(line.strip())
        f.close()
    return np.asarray(labels)


def data_generator(config_path,
                   label_encoder,
                   class_num,
                   batch_size,
                   input_shape=None,
                   label_parser=lambda label: label):

    filename_label_pairs = []
    with open(config_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            file_path, label = line.strip().split("\t")
            filename_label_pairs.append((file_path, label))
        f.close()

    batch_num = math.ceil(len(filename_label_pairs) / batch_size)

    i = -1
    while True:
        i += 1
        i %= batch_num

        filename_label_batch = filename_label_pairs[i * batch_size: (i + 1) * batch_size]

        print(filename_label_batch.__len__(), i * batch_size, (i + 1) * batch_size)

        imgs_batch = []
        labels_batch = []

        for file_path, label in filename_label_batch:

            image = cv2.imread(file_path)

            if input_shape is not None:
                image = cv2.resize(image, input_shape)

            image = np.asarray(image, dtype="float32") / 255.0

            imgs_batch.append(image)

            labels_batch.append(label)

        encoded_labels_array = label_encoder.transform(labels_batch).astype(np.int32)
        ohr_y_train = np_utils.to_categorical(encoded_labels_array, class_num)

        yield np.asarray(imgs_batch), ohr_y_train




def data_hybrid_generator(config_path,
                          gender_label_encoder,
                          birth_label_encoder,
                          gender_class_num,
                          birth_class_num,
                          batch_size,
                          input_shape=None,
                          label_parser=lambda label: label.split("_")):

    filename_label_pairs = []
    with open(config_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            filepath, label = line.strip().split("\t")
            filename_label_pairs.append((filepath, label_parser(label)))
        f.close()

    batch_num = math.ceil(len(filename_label_pairs) / batch_size)

    i = -1
    while True:
        i += 1
        i %= batch_num

        filename_label_batch = filename_label_pairs[i * batch_size: (i + 1) * batch_size]

        print(" ", i, filename_label_batch.__len__(), i * batch_size, (i + 1) * batch_size)
        # print(filename_label_batch)

        imgs_array = []
        gender_labels_array = []
        age_labels_array = []

        for filepath, label in filename_label_batch:

            image = io.imread(filepath)

            if input_shape is not None:
                image = transform.resize(image, input_shape, mode='constant')

            image = np.asarray(image, dtype="float32") / 255.0

            imgs_array.append(image)

            gender_labels_array.append(label[0])
            age_labels_array.append(label[1])

        encoded_gender_labels_array = gender_label_encoder.transform(gender_labels_array).astype(np.int32)
        ohr_gender_y_train = np_utils.to_categorical(encoded_gender_labels_array, gender_class_num)

        encoded_age_labels_array = birth_label_encoder.transform(age_labels_array).astype(np.int32)
        ohr_birth_y_train = np_utils.to_categorical(encoded_age_labels_array, birth_class_num)

        ohr_y_train = []

        for j in range(len(ohr_gender_y_train)):
            ohr_y_train.append(np.append(ohr_gender_y_train[j], ohr_birth_y_train[j]))

        yield np.asarray(imgs_array), np.asarray(ohr_y_train)












def load_data_from_config(path, input_shape=None, most=-1):

    X = []
    Y = []

    with open(path, "r") as f:

        i = 0
        for line in f:
            if line.strip() == "":
                continue

            if most > 0 and i > most:
                break
            i += 1

            filepath, label = line.strip().split("\t")

            image = cv2.imread(filepath)

            if input_shape is not None:
                image = cv2.resize(image, input_shape)

            image = np.asarray(image, dtype="float32") / 255.0
            X.append(image)
            Y.append(label)

        f.close()

    return np.asarray(X), np.asarray(Y)


def load_data(path):

    X = []
    Y = []

    gender_map = {}
    birth_map = {}

    i = 0
    j = 0
    for label in os.listdir(path):

        gender, birth = label.split("_")

        if gender not in gender_map:
            gender_map[gender] = i
            i += 1

        if birth not in birth_map:
            birth_map[birth] = j
            j += 1

    label_index_map = {}

    for gender, index in gender_map.items():
        label_index_map[gender] = index

    for birth, index in birth_map.items():
        label_index_map[birth] = index + len(gender_map)


    for label in os.listdir(path):

        for img_file in os.listdir(path + label):

            img = io.imread(path + label + "/" + img_file)
            X.append(img)

            y = np.zeros(label_index_map.__len__())

            gender, birth = label.split("_")

            y[label_index_map[gender]] = 1
            y[label_index_map[birth]] = 1

            Y.append(y)

    return np.asarray(X), np.asarray(Y)


