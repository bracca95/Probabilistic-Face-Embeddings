# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/
#py_objdetect/py_face_detection/py_face_detection.html

import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm

parser = argparse.ArgumentParser('align celeba face')
parser.add_argument('--input_dir', type=str, help='dir original dataset')
parser.add_argument('--output_dir', type=str, help='dir aligned images (out)')
parser.add_argument('--shape', type=int, default=128, help='final shape (square)')
args = parser.parse_args()

if __name__=='__main__':
    
    ## LOAD CASCADES (FRONTAL + PROFILE)
    path_to_cascade = '/usr/local/lib/python3.6/dist-packages/cv2/data'
    cascade_frontal = 'haarcascade_frontalface_default.xml'
    cascade_profile = 'haarcascade_profileface.xml'

    path_cascade_frontal = join(path_to_cascade, cascade_frontal)
    path_cascade_profile = join(path_to_cascade, cascade_profile)

    face_cascade_front = cv2.CascadeClassifier(path_cascade_frontal)
    face_cascade_profi = cv2.CascadeClassifier(path_cascade_profile)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ## READ IMAGES
    for i, elem in tqdm(enumerate(os.listdir(args.input_dir))):
        img_name = elem

        img_path_og = join(args.input_dir, img_name)
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
            shape = (128, 128)
            img_resz = cv2.resize(img, shape)


        ## OUTPUT
        img_out_path = join(args.output_dir, img_name)
        cv2.imwrite(img_out_path, img_resz)

        # # debug
        # if i%100==0:
        #     print(img_out_path)