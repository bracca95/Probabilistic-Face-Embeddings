# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/
#py_objdetect/py_face_detection/py_face_detection.html

import cv2
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm

def processing(fullDF, output_dir, csv_path):

    csv_path = os.path.abspath(os.path.expanduser(csv_path))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # check if there images are already present
    # https://stackoverflow.com/a/33400758
    if not any(fname.endswith('.jpg') for fname in os.listdir(output_dir)):

        df_label_list = list(fullDF['label'].unique())
        df_new = pd.DataFrame([], columns=['path', 'abspath', 'label', 'name'])
        
        for i, val in enumerate(tqdm(df_label_list)):
            filt = (fullDF['label'] == val)
            df_pair = fullDF.loc[filt]

            # full path
            img_bench_path = df_pair.iloc[0]['abspath']
            img_modif_path = df_pair.iloc[1]['abspath']

            # name
            img_bench_name = os.path.basename(img_bench_path)
            img_modif_name = os.path.basename(img_modif_path)

            # output path
            img_bench_out_path = join(output_dir, img_bench_name)
            img_modif_out_path = join(output_dir, img_modif_name)

            # new df
            static_img1 = df_pair.iloc[0][['path', 'label', 'name']]
            static_img1['abspath'] = img_bench_out_path
            static_img1 = static_img1[['path', 'abspath', 'label', 'name']]

            static_img2 = df_pair.iloc[1][['path', 'label', 'name']]
            static_img2['abspath'] = img_modif_out_path
            static_img2 = static_img2[['path', 'abspath', 'label', 'name']]

            df_new = df_new.append(static_img1, ignore_index=True)
            df_new = df_new.append(static_img2, ignore_index=True)

            write_img(img_bench_path, img_bench_out_path, upscale=False)
            write_img(img_modif_path, img_modif_out_path, upscale=True)

        df_new.to_csv(csv_path)
        return df_new



def write_img(img_path, output_path, upscale):

    img_colo = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_colo, cv2.COLOR_BGR2GRAY)

    if upscale == False:

        rows, cols, _ = map(int, img_colo.shape)

        old_shape = (rows, cols)    # h (row), w (col)
        new_shape = (112, 96)       # h (row), w (col)
        img = center_crop(img_colo, old_shape, new_shape)

        # normalization
        img_template = np.zeros(new_shape)
        img_norm = cv2.normalize(img, img_template, 0, 255, cv2.NORM_MINMAX)

        ## OUTPUT
        cv2.imwrite(output_path, img_norm)
    
    else:
        rows, cols, _ = map(int, img_colo.shape)

        old_shape = (rows, cols)
        new_shape = (112, 96)
        hr_shape = (128, 128)
        lr_shape = (16, 16)
        nTimes = int(math.log2(hr_shape[0]/lr_shape[0]))

        img = center_crop(img_colo, old_shape, hr_shape)

        # normalization
        img_template = np.zeros(hr_shape)
        img_norm = cv2.normalize(img, img_template, 0, 255, cv2.NORM_MINMAX)

        # resize to (16, 16)
        img_resz_lr = cv2.resize(img_norm, lr_shape)
        row, col, _ = map(int, img_resz_lr.shape)

        # upscale to (128, 128)
        for i in range(nTimes):
            img_resz_hr = cv2.pyrUp(img_resz_lr, dstsize=(2*row, 2*col))

            # update values
            img_resz_lr = img_resz_hr
            row, col, _ = map(int, img_resz_lr.shape)

        # crop to 112, 96
        img_fin = center_crop(img_resz_hr, hr_shape, new_shape)

        cv2.imwrite(output_path, img_fin)


# https://progr.interplanety.org/en/python-how-to-find-the-polygon-center-coordinates/
def center_crop(img, old_shape, new_shape):
    assert type(old_shape)==tuple, 'shape must be of type tuple'
    assert type(new_shape)==tuple, 'shape must be of type tuple'

    p1 = (0, 0)
    p2 = (0, old_shape[1])
    p3 = (old_shape[0], old_shape[1])
    p4 = (old_shape[0], 0)

    vertexes = (p1, p2, p3, p4)

    x_list = [vertex [0] for vertex in vertexes]
    y_list = [vertex [1] for vertex in vertexes]
    n_vert = len(vertexes)
    
    x = sum(x_list) / n_vert
    y = sum(y_list) / n_vert

    cent = (x, y)

    # upper left point cx = w + dw/2, cy = h + dh/2
    ulp_h = math.floor(cent[0] - (new_shape[0]/2))
    ulp_w = math.floor(cent[1] - (new_shape[1]/2))

    img = img[ulp_h:ulp_h+new_shape[0], ulp_w:ulp_w+new_shape[1]]
    
    return img