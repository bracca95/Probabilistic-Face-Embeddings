# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/
#py_objdetect/py_face_detection/py_face_detection.html

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm

def pre_processing(args):

    ## LOAD CASCADES (FRONTAL + PROFILE)
    path_to_cascade = '/usr/local/lib/python3.6/dist-packages/cv2/data'
    cascade_frontal = 'haarcascade_frontalface_default.xml'
    cascade_profile = 'haarcascade_profileface.xml'

    path_cascade_frontal = join(path_to_cascade, cascade_frontal)
    path_cascade_profile = join(path_to_cascade, cascade_profile)

    face_cascade_front = cv2.CascadeClassifier(path_cascade_frontal)
    face_cascade_profi = cv2.CascadeClassifier(path_cascade_profile)

    final_image_list = []

    ## READ IMAGES
    for i, elem in enumerate(tqdm(os.listdir(args.dataset_path))):
        img_name = elem

        img_path_og = join(args.dataset_path, img_name)
        img = cv2.imread(img_path_og, cv2.IMREAD_COLOR)

        img_colo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # use frontal is found, profile otherwise
        if face_cascade_front.detectMultiScale(img_gray, 1.3, 5) is not None:
            method = face_cascade_front.detectMultiScale(img_gray, 1.3, 5)
        else:
            method = face_cascade_profi.detectMultiScale(img_gray, 1.3, 5)

        for (x,y,w,h) in method:
            """ detect face

            force to be a square (in order to crop at 128x1218)
            chooose the bigger (h/w)"""
            
            l = h if h >= w else w
            img = img_colo[y:y+l, x:x+l]

            # resize
            shape_img = (96, 112)
            img_resz = cv2.resize(img, shape_img)

            # normalization
            img_template = np.zeros(shape_img)
            img_norm = cv2.normalize(img_resz, img_template, 0, 255, cv2.NORM_MINMAX)

            final_image_list.append(img_norm)


        ## OUTPUT
        return final_image_list

        # img_out_path = join(args.output_dir, img_name)
        # cv2.imwrite(img_out_path, img_resz)

        # # debug
        # if i%100==0:
        #     print(img_out_path)