# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np

from network.network import Network

np.random.seed(666)


class LeNet5(Network):

    def build_network(self, input_shape, output_shape):
        self.model = Sequential()
        self.model.add(Conv2D(32, (5, 5), strides=(1, 1), input_shape=input_shape, padding='valid', activation='relu',
                         kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(output_shape, activation='softmax'))
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

