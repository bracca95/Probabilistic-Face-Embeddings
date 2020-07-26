# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/
#py_objdetect/py_face_detection/py_face_detection.html

import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm

def processing(fullDF, output_dir):

    ## LOAD CASCADES (FRONTAL + PROFILE)
    path_to_cascade = '/usr/local/lib/python3.6/dist-packages/cv2/data'
    cascade_frontal = 'haarcascade_frontalface_default.xml'
    cascade_profile = 'haarcascade_profileface.xml'

    path_cascade_frontal = join(path_to_cascade, cascade_frontal)
    path_cascade_profile = join(path_to_cascade, cascade_profile)

    face_cascade_front = cv2.CascadeClassifier(path_cascade_frontal)
    face_cascade_profi = cv2.CascadeClassifier(path_cascade_profile)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## READ IMAGES
    for i in tqdm(range(len(fullDF))):
        img_path_og = fullDF.iloc[i]['abspath']
        img_name = os.path.basename(img_path_og)
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


        ## OUTPUT
        img_out_path = join(output_dir, img_name)
        cv2.imwrite(img_out_path, img_resz)

        ## REPLACE ABSPATHS
        fullDF.iloc[i]['abspath'] = img_out_path

        # # debug
        # if i%100==0:
        #     print(img_out_path)

    return fullDF