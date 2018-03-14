# -*- coding: utf-8 -*-

import data.dataset as dataset
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from network.xception import Xception

def run_hybrid_xception():

    genders = dataset.load_labels("../data/gender_label.txt")
    births = dataset.load_labels("../data/age_label.txt")
    x_test, y_test = dataset.load_data_from_config("../data/hybrid_test.txt", input_shape=(299, 299))

    gender_y_test = [y.split("_")[0] for y in y_test]
    birth_y_test = [y.split("_")[1] for y in y_test]

    gender_encoder = LabelEncoder()
    gender_encoder.fit(genders)
    encoded_gender_y_test = gender_encoder.transform(gender_y_test).astype(np.int32)
    one_hot_gender_y_test = np_utils.to_categorical(encoded_gender_y_test, np.max(encoded_gender_y_test) + 1)

    birth_encoder = LabelEncoder()
    birth_encoder.fit(births)
    encoded_birth_y_test = birth_encoder.transform(birth_y_test).astype(np.int32)
    one_hot_birth_y_test = np_utils.to_categorical(encoded_birth_y_test, np.max(encoded_birth_y_test) + 1)

    hybrid_y_test = []
    for i in range(len(one_hot_gender_y_test)):
        hybrid_y_test.append(np.append(one_hot_gender_y_test[i], one_hot_birth_y_test[i]))
    hybrid_y_test = np.asarray(hybrid_y_test)



    # ohr_Y_train = np_utils.to_categorical(encoded_age_Y_train, np.max(encoded_age_Y_train) + 1)
    # ohr_Y_test = np_utils.to_categorical(encoded_Y_test, np.max(encoded_Y_test) + 1)

    network = Xception()

    network.build_network(include_top=True, input_shape=(299, 299, 3), output_shape=hybrid_y_test[0].__len__())

    network.run_generator(dataset.data_hybrid_generator("../data/hybrid_train.txt",
                                                        gender_encoder,
                                                        birth_encoder,
                                                        genders.__len__(),
                                                        births.__len__(),
                          1),
                          loss="categorical_crossentropy",
                          optimizer="sgd",
                          steps_per_epoch=8,
                          val_data=(x_test, hybrid_y_test))  # (X_test, ohr_Y_test)

if __name__ == "__main__":

    run_hybrid_xception()