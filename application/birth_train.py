# -*- coding: utf-8 -*-
import keras
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import prepare.dataset as dataset
from network.mobilenet import MobileNet

birth_year = [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]


def train_mobilenet_with_data_generator(batch_size, input_shape):

    class_num = birth_year.__len__()

    filepath = '../models/birth/birth.mobilenet.{epoch:02d}-{loss:.2f}.model'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=batch_size, write_graph=True,
                                              write_grads=False,
                                              write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                              embeddings_metadata=None)

    X_test, Y_test = dataset.load_data_from_config("../data/birth/birth_test.txt")

    encoder = LabelEncoder()
    encoder.fit(birth_year)
    encoded_Y_test = encoder.transform(Y_test).astype(np.int32)

    ohr_Y_test = np_utils.to_categorical(encoded_Y_test, class_num)

    generator = dataset.data_generator("../data/birth/birth_train.txt",
                                       encoder,
                                       class_num,
                                       batch_size=batch_size)


    net = MobileNet()
    net.build_network(include_top=True, input_shape=(224, 224, 3), output_shape=class_num)
    net.run_generator(
        generator,
        loss="categorical_crossentropy",
        optimizer="sgd",
        epoch=1,
        steps_per_epoch=10,
        val_data=(X_test, ohr_Y_test),
        callbacks=[checkpoint, tensorboard]
    )

    net.save("../model/birth/birth.mobilenet.h5")


if __name__ == "__main__":
    train_mobilenet_with_data_generator()

