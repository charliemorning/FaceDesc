# -*- coding: utf-8 -*-

import keras
import numpy as np
import data.dataset as dataset
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from network.mobilenet import MobileNet


def train_mobilenet_with_data_generator():

    filepath = '../emotion/emotion.mobilenet.{epoch:02d}-{loss:.2f}.model'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=8, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)

    X_test, Y_test = dataset.load_data_from_config("../data/emotion/emotion_test.txt", input_shape=(224, 224))

    encoder = LabelEncoder()
    encoder.fit(["0", "1", "2", "3", "4", "5", "6"])
    encoded_Y_test = encoder.transform(Y_test).astype(np.int32)

    ohr_Y_test = np_utils.to_categorical(encoded_Y_test, 7)

    generator = dataset.data_generator("../data/emotion/emotion_train.txt",
                                       encoder,
                                       7,
                                       input_shape=(224, 224),
                                       batch_size=20)

    net = MobileNet()
    net.build_network(include_top=True, input_shape=(224, 224, 3), output_shape=7)
    net.run_generator(
        generator,
        loss="categorical_crossentropy",
        optimizer="sgd",
        epoch=2,
        steps_per_epoch=10,
        val_data=(X_test, ohr_Y_test),
        callbacks=[checkpoint, tensorboard]
    )

    net.save("../models/emotion/emotion.mobilenet.final.model")


if __name__ == '__main__':
    train_mobilenet_with_data_generator()