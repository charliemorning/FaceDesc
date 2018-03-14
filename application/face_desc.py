# -*- coding: utf-8 -*-


import datetime

import cv2
import numpy as np

from application.birth_predict import predict_birth
from application.face_search_predict import search_face
from application.gender_predict import predict_gender
from util.align_dataset_mtcnn import face_detect
from util.face_repr import get_face_repr


def predict_age(image):
    birth = predict_birth(image)
    return datetime.datetime.now().year - birth - np.random.randint(-3, 3)

def main():
    print("begin...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("face description", frame)

        key = cv2.waitKey(10)

        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord(' '):
            faces = face_detect(frame)
            for face in faces:
                # face = cv2.flip(face, 1)
                gender = predict_gender(face)
                age = predict_age(face)

                vect = get_face_repr(face)

                name = str([i for i in search_face(vect)])

                text = gender + " " + str(age) + " " + name

                print(text)


if __name__ == "__main__":
    main()




