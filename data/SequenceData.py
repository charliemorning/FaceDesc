# -*- coding: utf-8 -*-

import glob
import math
import os

import numpy as np
from keras.utils import Sequence
from keras.utils import np_utils
from skimage import io, transform



class SequenceData(Sequence):

    def __init__(self, path, batch_size, input_shape, label_encoder, data_format="png"):
        self.path = path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.label_encoder = label_encoder
        self.data_format = data_format

        self.filename_label_list = []

        with open(path, "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                filepath, label = line.strip().split("\t")

                self.filename_label_list.append((filepath, label.strip()))

        # for label in os.listdir(path):
        #     self.filename_label_list.extend([(filename, label) for filename in glob.glob(self.path + "*." + data_format)])

    def __len__(self):
        num_imgs = len(glob.glob(self.path + '*.' + self.data_format))
        return math.ceil(num_imgs / self.batch_size)

    def __getitem__(self, idx):

        batch = self.filename_label_list[idx * self.batch_size: (idx + 1) * self.batch_size]

        print(idx)
        print(idx * self.batch_size,  (idx + 1) * self.batch_size)

        x_train = []
        y_train = []
        for filename, label in batch:
            x_train.append(self.__read_img(filename))
            y_train.append(label)

        x_train = np.asarray(x_train)

        encoded_y_train = self.label_encoder.transform(y_train).astype(np.int32)
        ohr_y_train = np_utils.to_categorical(encoded_y_train, np.max(encoded_y_train) + 1)

        return x_train, ohr_y_train

    def __read_img(self, x):
        print(x)
        try:
            img = io.imread(x)
            img = transform.resize(img, self.input_shape)
        except Exception as e:
            print(e)
        else:
            return img