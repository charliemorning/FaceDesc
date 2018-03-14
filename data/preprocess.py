# -*- coding: utf-8 -*-

import tarfile
import os
import shutil
from skimage import io, transform

def image_file(members):
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[0].endswith("S"):
            yield tarinfo


def get_gender_from_id_card_no(id_card_no):
    if int(id_card_no[16]) % 2 == 0:
        return "M"
    else:
        return "F"


def get_birth_year_from_id_card_no(id_card_no):
    return id_card_no[6:10]


def untar_stm_image_file_tarball(path, dst_path):
    tar = tarfile.open(path)
    tar.extractall(path=dst_path, members=image_file(tar))
    tar.close()


def process_all_tarball(stm_tarfile_root_path, dst_path):

    for id_card_no in os.listdir(stm_tarfile_root_path):
        if id_card_no.__len__() != 18:
            continue

        gender = get_gender_from_id_card_no(id_card_no)
        birth_year = get_birth_year_from_id_card_no(id_card_no)

        label = gender + "_" + birth_year

        if not os.path.exists(dst_path + label):
            os.mkdir(dst_path + label)

        for tarball in os.listdir(stm_tarfile_root_path + id_card_no):
            if not tarball.endswith("tar"):
                continue
            try:
                untar_stm_image_file_tarball(stm_tarfile_root_path + id_card_no + "/" + tarball, dst_path + label)
            except:
                print(id_card_no)


def relocate_all_image(dst_path):

    for label in os.listdir(dst_path):
        for dir in os.listdir(dst_path + label):
            if not os.path.isdir(dst_path + label + "/" + dir):
                continue
            for f in  os.listdir(dst_path + label + "/" + dir):
                shutil.move(dst_path + label + "/" + dir + "/" + f, dst_path + label + "/" + f)
            shutil.rmtree(dst_path + label + "/" + dir)


def resize_all(src_path, dst_path, shape):

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    for dir in os.listdir(src_path):

        if not os.path.exists(dst_path + dir):
            os.mkdir(dst_path + dir)

        for image_name in os.listdir(src_path + dir):

            image = io.imread(src_path + dir + "/" + image_name)

            image = transform.resize(image, shape)

            io.imsave(dst_path + dir + "/" + image_name, image)


if __name__ == "__main__":
    path = "H:/zc/stm/"
    # process_all_tarball("H:/zc/facedataset/", "H:/zc/facedataset_20180213/")
    relocate_all_image("H:/zc/facedataset_20180213/")
