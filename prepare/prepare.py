# -*- coding: utf-8 -*-

import csv
import os
from collections import Counter

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def write_file(x, y, filename):
    with open(filename, "w") as f:
        for i in range(x.__len__()):
            f.write(x[i])
            f.write("\t")
            f.write(y[i])
            f.write("\n")
        f.close()


def write_label_file(labels, filename):

    with open(filename, "w") as f:
        for label in labels:
            f.write(label)
            f.write("\n")
        f.close()


def prepare_gender_birth_train_test_meta_file(path):

    full_path_list = []
    gender_labels_list = []
    birth_labels_list = []
    for label in os.listdir(path):
        for img_file in os.listdir(path + label):

            full_path = path + label + "/" + img_file
            gender, birth = label.split("_")

            birth = str((int(birth[:-1]) + 1) * 10)

            full_path_list.append(full_path)
            gender_labels_list.append(gender)
            birth_labels_list.append(birth)

    gender_x_train, gender_x_test, gender_y_train, gender_y_test =\
        train_test_split(full_path_list, gender_labels_list, test_size=0.01, random_state=666)

    birth_x_train, birth_x_test, birth_y_train, birth_y_test = \
        train_test_split(full_path_list, birth_labels_list, test_size=0.01, random_state=666)

    write_file(gender_x_train, gender_y_train, "gender_train.txt")
    write_file(gender_x_test, gender_y_test, "gender_test.txt")
    write_file(birth_x_train, birth_y_train, "birth_train.txt")
    write_file(birth_x_test, birth_y_test, "birth_test.txt")

def prepare_train_test_meta_file(path, train_file, test_file, test_size=0.01, label_proc=lambda label: label):

    full_path_list = []
    labels_list = []
    for label in os.listdir(path):
        for img_file in os.listdir(path + label):
            full_path = path + label + "/" + img_file
            full_path_list.append(full_path)
            labels_list.append(label_proc(label))

    x_train, x_test, y_train, y_test =\
        train_test_split(full_path_list, labels_list, test_size=test_size, random_state=666)

    write_file(x_train, y_train, train_file)
    write_file(x_test, y_test, test_file)


def prepare_hybrid_class(path):
    full_path_list = []
    labels_list = []
    gender_labels_list = []
    birth_labels_list = []
    for label in os.listdir(path):
        for img_file in os.listdir(path + label):
            full_path = path + label + "/" + img_file
            full_path_list.append(full_path)
            labels_list.append(label)
            gender, birth = label.split("_")
            gender_labels_list.append(gender)
            birth_labels_list.append(birth)

    x_train, x_test, y_train, y_test = \
        train_test_split(full_path_list, labels_list, test_size=0.1, random_state=666)

    write_file(x_train, y_train, "hybrid_train.txt")
    write_file(x_test, y_test, "hybrid_test.txt")
    write_label_file(set(gender_labels_list), "gender_label.txt")
    write_label_file(set(birth_labels_list), "age_label.txt")




def prepare_fer2013(fer2013_path, dst_path):

    emotion_counter = Counter(["0", "1", "2", "3", "4", "5", "6"])

    with open(fer2013_path, "r") as f:
        reader = csv.DictReader(f)
        for obj in reader:
            usage = obj["Usage"]
            emotion = obj["emotion"]
            pixels = obj["pixels"]

            if usage != "Training":
                continue

            if not os.path.exists(dst_path + emotion):
                os.mkdir(dst_path + emotion)

            emotion_counter[emotion] += 1

            image = np.asarray([float(p) for p in pixels.split()]).reshape(48, 48)

            image_filename = str(emotion_counter[emotion]) + ".png"

            cv2.imwrite(dst_path + emotion + "/" + image_filename, image)


def extract_face_repr(src_path, dst_path):
    from util.face_repr import get_face_repr
    X = []
    Y = []
    for label in os.listdir(src_path):
        for image_file_name in os.listdir(src_path + label):
            image = cv2.imread(src_path + label + "/" + image_file_name)
            vect = get_face_repr(image)
            X.append(vect[0])
            Y.append(label)

    with open(dst_path, "w") as f:
        for i in range(X.__len__()):
            f.write(" ".join([str(v) for v in X[i]]))
            f.write(",")
            f.write(Y[i])
            f.write("\n")


if __name__ == "__main__":
    # prepare_gender_birth_train_test_meta_file("C:/developer/dataset/gender_test/train/")
    # prepare_train_test_meta_file("C:/developer/dataset/emotion/", "emotion_train.txt", "emotion_test.txt")
    # pass
    # prepare_hybrid_class("C:/developer/dataset/gender_test/train/")

    # prepare_fer2013("../data/emotion/fer2013.csv", "C:/developer/dataset/emotion/")

    # extract_face_repr("image/face_recognition/", "face_recognition_face_rep.txt")










