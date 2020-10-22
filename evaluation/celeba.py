"""Test protocols on LFW dataset
"""
# MIT License
# 
# Copyright (c) 2017 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation import metrics
from collections import namedtuple

StandardFold = namedtuple('StandardFold', ['indices1', 'indices2', 'labels'])

class CelebATest:
    def __init__(self, image_paths):
        # kind of image paths list
        self.image_paths = np.array(image_paths).astype(np.object).flatten()
        self.images = None
        self.labels = None
        self.standard_folds = None
        self.blufr_folds = None
        self.queue_idx = None
        self.n_fold = 10

    def init_standard_proto(self, lfw_pairs_file):

        self.standard_folds = []

        self.index_dict = pd.DataFrame([], columns=['key', 'val'])
        for i, image_path in enumerate(self.image_paths):
            image_name, image_ext = os.path.splitext(os.path.basename(image_path))
            line = pd.DataFrame([[image_name, i]], columns=['key', 'val'])
            self.index_dict = self.index_dict.append(line)

        # self.index_dict = {}
        # for i, image_path in enumerate(self.image_paths):
        #     image_name, image_ext = os.path.splitext(os.path.basename(image_path))
        #     self.index_dict[image_name] = i

        # retrieve all the IDs (unique)
        df = lfw_pairs_file
        id_list = list(df['label'].unique())
        
        rang = 600
        rang_half = int(rang/2)

        for i in range(self.n_fold):

            # init lists
            indices1 = np.zeros(rang, dtype=np.int32)
            indices2 = np.zeros(rang, dtype=np.int32)
            labels = np.array([True]*rang_half+[False]*rang_half, dtype=np.bool)

            for iden in range(rang):

                # retrieve an image label
                tot_iter = (rang*i) + (iden)        # overall iteration
                curr_id = id_list[tot_iter]         # ID at iteration tot_iter
                filt = (df['label'] == curr_id)     # filter for that ID
                
                # filter two different DFs
                df_pair = df.loc[filt]
                
                # https://www.geeksforgeeks.org/how-to-randomly-select-rows-from-pandas-dataframe/
                # two images with same ID, the third image is another person
                df_impo = df.loc[~filt]
                df_impostor_line = df_impo.sample()
                df_triple = df_pair.append(df_impostor_line)

                if iden < rang_half:
                    # get face numbers (image id)
                    face1 = df_triple['path'].iloc[0]   # face 1
                    face2 = df_triple['path'].iloc[1]   # face 2
                else:
                    # get face 1 + impostor
                    face1 = df_triple['path'].iloc[0]   # face 1
                    face2 = df_triple['path'].iloc[2]   # impostor
                
                f1, _ = os.path.splitext(face1)          # get string face 1
                f2, _ = os.path.splitext(face2)          # get string face 2

                filt_f1 = (self.index_dict['key'] == f1)
                filt_f2 = (self.index_dict['key'] == f2)

                indices1[iden] = self.index_dict.loc[filt_f1]['val'].values[0]
                indices2[iden] = self.index_dict.loc[filt_f2]['val'].values[0]

            fold = StandardFold(indices1, indices2, labels)
            self.standard_folds.append(fold)


    def test_standard_proto(self, features, compare_func, pos_idx):

        assert self.standard_folds is not None
        
        accuracies = np.zeros(self.n_fold, dtype=np.float32)
        thresholds = np.zeros(self.n_fold, dtype=np.float32)

        features1 = []
        features2 = []

        for i in range(self.n_fold):
            """
            The network needs training: select all the j (600) images for each
            folder apart from the one that, for each iteration, have the same
            value of i (e.g. at iteration 23 -> i=23, skip identity j=23).
            All of those that are skipped will be used for testing (one folder
            - the remaining 600/6.000 values - for every iteration).
            """

            # Training
            train_indices1 = np.concatenate(\
                [self.standard_folds[j].indices1 for j in range(self.n_fold) if j!=i])
            train_indices2 = np.concatenate(\
                [self.standard_folds[j].indices2 for j in range(self.n_fold) if j!=i])
            train_labels = np.concatenate(\
                [self.standard_folds[j].labels for j in range(self.n_fold) if j!=i])

            # position[train_indices1, all_the_512_features]
            train_features1 = features[train_indices1,:]
            train_features2 = features[train_indices2,:]
            
            train_score = compare_func(train_features1, train_features2)
            _, thresholds[i] = metrics.accuracy(train_score, train_labels)

            # Testing
            fold = self.standard_folds[i]
            test_features1 = features[fold.indices1,:]
            test_features2 = features[fold.indices2,:]
            test_labels = fold.labels
            
            test_score = compare_func(test_features1, test_features2)
            accuracies[i], _ = metrics.accuracy(test_score, test_labels, np.array([thresholds[i]]))

            """se i == 0 prendi uno degli index a caso tra i 600 e risali al
            nome delle immagini tramite self.index_dict, così dopo le puoi plottare
            """
            if i == 0:
                pos_idx
                lab = True if pos_idx < 300 else False

                img1 = fold.indices1[pos_idx]
                img2 = fold.indices2[pos_idx]

                feat1 = features[img1, :]
                feat2 = features[img2, :]

                # shape depends on compare function 1st (1, 512) / 2nd (1, 1024)
                feat1 = feat1.reshape(1, -1)
                feat2 = feat2.reshape(1, -1)

                score = compare_func(feat1, feat2)
                #accur, _ = metrics.accuracy(score, lab, np.array([thresholds[i]]))

                accur = np.zeros(1)
                pred_vec = score>=thresholds[i]
                accur = np.mean(pred_vec==lab)

                # retrieve image name
                filt_img1 = (self.index_dict['val'] == img1)
                filt_img2 = (self.index_dict['val'] == img2)
                image_name1 = self.index_dict.loc[filt_img1]['key'].values[0]
                image_name2 = self.index_dict.loc[filt_img2]['key'].values[0]

                print(f'image1:{image_name1},\
                        image2:{image_name2},\
                        label:{lab},\
                        score:{score},\
                        accuracy:{accur}\n')

                img_fold = '/content/Probabilistic-Face-Embeddings/data/celeba_align'

                plt.figure()
                f, axarr = plt.subplots(1,2)
                f.suptitle(f'same={str(lab)}, accuracy={str(accur)}')
                
                i1 = cv2.imread(os.path.join(img_fold, f'{image_name1}.jpg'), cv2.IMREAD_COLOR)
                i2 = cv2.imread(os.path.join(img_fold, f'{image_name2}.jpg'), cv2.IMREAD_COLOR)
                
                i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
                i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2RGB)
                axarr[0].imshow(i1)
                axarr[1].imshow(i2)

                save_fold = '/content/out_figures'
                if not os.path.exists(save_fold):
                    os.makedirs(save_fold)
                plt.savefig(os.path.join(save_fold, f'image_{pos_idx}.jpg'))


        accuracy = np.mean(accuracies)
        threshold = - np.mean(thresholds)
        return accuracy, threshold

