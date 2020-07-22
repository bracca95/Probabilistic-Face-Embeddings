"""Data fetching with pandas
"""
# MIT License
# 
# Copyright (c) 2018 Yichun Shi
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

import sys
import os
import time
import math
import random
import shutil
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd

queue_timeout = 600

class Dataset(object):

    def __init__(self, path=None, prefix=None):

        if path is not None:
            self.init_from_path(path)
        else:
            # empty dataset
            self.data = pd.DataFrame([], columns=['path', 'abspath', 'label', 'name'])

        self.prefix = prefix
        self.base_seed = 0
        self.batch_queue = None
        self.batch_workers = None
       

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        return self.data[key]

    def _delitem(self, key):
        self.data.__delitem__(key)

    @property
    def num_classes(self):
        return len(self.data['label'].unique())

    @property
    def classes(self):
        return self.data['label'].unique()

    @property
    def size(self):
        return self.data.shape[0]

    @property
    def loc(self):
        return self.data.loc       

    @property
    def iloc(self):
        return self.data.iloc

    def init_from_path(self, path):
        """ method

        if the passed string is a folder, go to function init_from_folder
        if the passed string is a .txt file, go to init_from_list
        """
        path = os.path.expanduser(path)
        _, ext = os.path.splitext(path)
        
        # if path is a folder
        if os.path.isdir(path):
            #check if there are subfolders or only files (images)
            for _, dirs, _ in os.walk(path):
                pass
            
            # if only files
            if not dirs:
                self.init_from_images(path)
            # if there are subfolders
            else:
                self.init_from_folder(path)
        
        elif ext == '.txt':
            self.init_from_list(path)
        else:
            raise ValueError('Cannot initialize dataset from path: %s\n\
                It should be either a folder, .txt or .hdf5 file' % path)
        # print('%d images of %d classes loaded' % (len(self.images), self.num_classes))


    def init_from_images(self, folder):
        """idea: use your method in PFSR to get a dataframe with paths, abspath,
        labels (identity) and names (as 'celebName_label')
        """
        folder = os.path.abspath(os.path.expanduser(folder))
        csv_partition = 'list_eval_partition.csv'
        csv_identity = 'identity_CelebA.csv'

        csv_partition_path = os.path.join(os.path.dirname(folder), csv_partition)
        csv_identity_path = os.path.join(os.path.dirname(folder), csv_identity)

        df_partition = pd.read_csv(csv_partition_path)
        df_identity = pd.read_csv(csv_identity_path)

        df = df_partition.copy()

        # this can be done because indexing is the same 
        # https://stackoverflow.com/a/45747631
        # now we have a dataset with ['image_id']['partition']['identity']
        df['identity'] = df_identity['identity']
        
        # format dataframe as the program requires
        # https://stackoverflow.com/a/20027386
        # https://stackoverflow.com/a/53816799
        df_full = pd.DataFrame({
            'path': df['image_id'],
            'abspath': folder + '/' + df['image_id'].astype(str),
            'label': df['identity'],
            'name': df['identity']
            })

        ## GET ONLY TWO IMAGES FOR IDENTITY
        df_min = df.groupby(['identity']).min()
        df_min.reset_index(inplace=True)
        df_min = df_min[['image_id', 'partition', 'identity']]

        df_max = df.groupby(['identity']).max()
        df_max.reset_index(inplace=True)
        df_max = df_max[['image_id', 'partition', 'identity']]

        filt = (df['image_id'].isin(df_min['image_id']) == True) | \
               (df['image_id'].isin(df_max['image_id']) == True)
        df_compare = df.loc[filt]

        # return values the program requires
        self.data = df_compare
        self.prefix = folder



    def init_from_images(self, folder):
        # if dataset main folder is provided 
        folder = os.path.abspath(os.path.expanduser(folder))
        class_names = os.listdir(folder)    # ls subfolders
        class_names.sort()                  # sort subfolders by name
        
        paths = []
        labels = []
        names = []
        
        for label, class_name in enumerate(class_names):
            classdir = os.path.join(folder, class_name)     # full subdir name
            if os.path.isdir(classdir):
                # ls all images and get full name
                images_class = os.listdir(classdir)
                images_class.sort()
                images_class = [os.path.join(class_name,img) for img in images_class]
                
                # fill the lists
                paths.extend(images_class)
                labels.extend(len(images_class) * [label])  # assign label i to each image in asubdir
                names.extend(len(images_class) * [class_name])
        
        abspaths = [os.path.join(folder,p) for p in paths]
        self.data = pd.DataFrame({'path': paths, 'abspath': abspaths, 
                                            'label': labels, 'name': names})
        self.prefix = folder

    

    def init_from_list(self, filename, folder_depth=2):
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines]
        abspaths = [os.path.abspath(line[0]) for line in lines]
        paths = ['/'.join(p.split('/')[-folder_depth:]) for p in abspaths]
        if len(lines[0]) == 2:
            labels = [int(line[1]) for line in lines]
            names = [str(lb) for lb in labels]
        elif len(lines[0]) == 1:
            names = [p.split('/')[-folder_depth] for p in abspaths]
            _, labels = np.unique(names, return_inverse=True)
        else:
            raise ValueError('List file must be in format: "fullpath(str) \
                                        label(int)" or just "fullpath(str)"')

        self.data = pd.DataFrame({'path': paths, 'abspath': abspaths, 
                                            'label': labels, 'name': names})
        self.prefix = abspaths[0].split('/')[:-folder_depth]


    #
    # Data Loading
    #

    def set_base_seed(self, base_seed=0):
        self.base_seed = base_seed

    def random_samples_from_class(self, label, num_samples, exception=None):
        # indices_temp = self.class_indices[label]
        indices_temp = list(np.where(self.data['label'].values == label)[0])
        
        if exception is not None:
            indices_temp.remove(exception)
            assert len(indices_temp) > 0
        # Sample indices multiple times when more samples are required than present.
        indices = []
        iterations = int(np.ceil(1.0*num_samples / len(indices_temp)))
        for i in range(iterations):
            sample_indices = np.random.permutation(indices_temp)
            indices.append(sample_indices)
        indices = list(np.concatenate(indices, axis=0)[:num_samples])
        return indices

    def get_batch_indices(self, batch_format):
        ''' Get the indices from index queue and fetch the data with indices.'''
        indices_batch = []
        batch_size = batch_format['size']

        num_classes = batch_format['num_classes']
        assert batch_size % num_classes == 0
        num_samples_per_class = batch_size // num_classes
        idx_classes = np.random.permutation(self.classes)[:num_classes]
        indices_batch = []
        for c in idx_classes:
            indices_batch.extend(self.random_samples_from_class(c, num_samples_per_class))

        return indices_batch

    def get_batch(self, batch_format):

        indices = self.get_batch_indices(batch_format)
        batch = {}
        for column in self.data.columns:
            batch[column] = self.data[column].values[indices]

        return batch

    # Multithreading preprocessing images

    def start_batch_queue(self, batch_format, proc_func=None, maxsize=1, num_threads=3):

        self.batch_queue = Queue(maxsize=maxsize)
        def batch_queue_worker(seed):
            np.random.seed(seed+self.base_seed)
            while True:
                batch = self.get_batch(batch_format)
                if proc_func is not None:
                    batch['image'] = proc_func(batch['abspath'])
                self.batch_queue.put(batch)

        self.batch_workers = []
        for i in range(num_threads):
            worker = Process(target=batch_queue_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)
    
    def pop_batch_queue(self, timeout=queue_timeout):
        return self.batch_queue.get(block=True, timeout=timeout)
      
    def release_queue(self):
        if self.index_queue is not None:
            self.index_queue.close()
        if self.batch_queue is not None:
            self.batch_queue.close()
        if self.index_worker is not None:
            self.index_worker.terminate()   
            del self.index_worker
            self.index_worker = None
        if self.batch_workers is not None:
            for w in self.batch_workers:
                w.terminate()
                del w
            self.batch_workers = None

