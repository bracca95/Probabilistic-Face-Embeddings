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
import pandas as pd
import numpy as np
import scipy.io as sio
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

    def init_standard_proto(self, lfw_pairs_file):

        self.standard_folds = []
        df = lfw_pairs_file
        rang = 3000

        for it, iden in enumerate(range(rang)):
            filt = (df['identity'] == iden)
            
            # filter two different DFs
            df_pair = df.loc[filt]
            df_impo = df.loc[~filt]
            # https://www.geeksforgeeks.org/how-to-randomly-select-rows-from-pandas-dataframe/
            df_impostor_line = df_impo.sample()

            # two images with same ID, the third image is another person
            df_triple = df_pair.append(df_impostor_line)

            # init lists
            indices1 = np.zeros(rang, dtype=np.int32)
            indices2 = np.zeros(rang, dtype=np.int32)
            lab_list = np.array(([True]+[False])*rang, dtype=np.bool)

            # get face numbers (image id)
            face1 = df_triple['image_id'].iloc[0]   # face 1
            face2 = df_triple['image_id'].iloc[1]   # face 2
            face3 = df_triple['image_id'].iloc[2]   # impostor
            
            f1, _ = os.path.splitext(face1)          # get integer face 1
            f2, _ = os.path.splitext(face2)          # get integer face 2
            f3, _ = os.path.splitext(face3)          # get integer impostor

            # position
            pos = 2*it
            
            # fill positive examples
            indices1[pos] = f1
            indices2[pos] = f2
            lab_list[pos] = True

            # fill negative examples
            indices1[pos+1] = f1
            indices2[pos+1] = f3
            lab_list[pos+1] = False

            fold = StandardFold(indices1, indices2, labels)
            self.standard_folds.append(fold)
            


    def test_standard_proto(self, features, compare_func):

        assert self.standard_folds is not None
        
        accuracies = np.zeros(10, dtype=np.float32)
        thresholds = np.zeros(10, dtype=np.float32)

        features1 = []
        features2 = []

        for i in range(10):
            # Training
            train_indices1 = np.concatenate([self.standard_folds[j].indices1 for j in range(10) if j!=i])
            train_indices2 = np.concatenate([self.standard_folds[j].indices2 for j in range(10) if j!=i])
            train_labels = np.concatenate([self.standard_folds[j].labels for j in range(10) if j!=i])

            train_features1 = features[train_indices1,:]
            train_features2 = features[train_indices2,:]
            
            train_score = compare_func(train_features1, train_features2)
            _, thresholds[i] = metrics.accuracy(train_score, train_labels)

            # Testing
            fold = self.standard_folds[i]
            test_features1 = features[fold.indices1,:]
            test_features2 = features[fold.indices2,:]
            
            test_score = compare_func(test_features1, test_features2)
            accuracies[i], _ = metrics.accuracy(test_score, fold.labels, np.array([thresholds[i]]))

        accuracy = np.mean(accuracies)
        threshold = - np.mean(thresholds)
        return accuracy, threshold

