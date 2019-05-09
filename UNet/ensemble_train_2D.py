from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

from image_util import BaseDataProvider



class Parameter(object):
    def __init__(self): 
        self.suffix = '5layer'
        self.root_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/ensemble/{}/result/pre_image'.format(self.suffix)
        self.channel=4
        self.layers=5
        self.features_root=32
        self.batch_size=4
        self.training_iters=117
        self.epochs=120
        self.display_step=117*8     #number of steps till outputting stats
        self.dropout=1.0
        self.restore=False       #Flag if previous model should be restored
        self.learning_rate=0.0001
        self.decay_rate=0.5
        self.decay_epochs=90
        self.mask=0.5

class ensembleProvider(BaseDataProvider):
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix="fat.npy",
                 mask_suffix='label.npy', shuffle_data=True, n_class = 2):
        super(ensembleProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        
        self.data_files = self._find_data_files(search_path)
        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))
        
        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if self.data_suffix in name]
    
    def _load_file(self, path, dtype=np.float32):
        fat_path = path.replace(self.data_suffix, "fat.npy")
        inn_path = path.replace(self.data_suffix, "inn.npy")
        wat_path = path.replace(self.data_suffix, "wat.npy")
        opp_path = path.replace(self.data_suffix, "opp.npy")
        fat_img = np.array(np.load(fat_path), dtype=dtype)
        inn_img = np.array(np.load(inn_path), dtype=dtype)
        wat_img = np.array(np.load(wat_path), dtype=dtype)
        opp_img = np.array(np.load(opp_path), dtype=dtype)

        img = np.zeros((fat_img.shape[0], fat_img.shape[1], 4), dtype=dtype)
        img[...,0] = fat_img
        img[...,1] = inn_img
        img[...,2] = wat_img
        img[...,3] = opp_img

        return img

    def _load_label(self, path, dtype=np.bool):
        return np.array(np.load(path), dtype=dtype) 

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.data_files)
        
    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        
        img = self._load_file(image_name, np.float32)
        label = self._load_label(label_name, np.bool)
    
        return img,label

    def _load_data_and_label(self):
        data, label = self._next_data()

        train_data = self._process_data(data)
        labels = self._process_labels(label)

        train_data, labels = self._post_process(train_data, labels)

        nx = train_data.shape[0]
        ny = train_data.shape[1]

        return train_data.reshape(1, nx, ny, self.channels), labels.reshape(1, ny, nx, self.n_class)

    def __call__(self, n):
        train_data, labels, range_arr = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
        Z = np.zeros((n, 8))

        X[0] = train_data
        Y[0] = labels
        Z[0] = range_arr
        for i in range(1, n):
            train_data, labels, range_arr = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
            Z[i] = range_arr

        return X, Y, Z


if __name__ == '__main__':
    para = Parameter()
    root_address = para.root_address
    generator_address = os.path.join(root_address, 'data/train/*/*/*.npy')
    generator = shapeProvider(generator_address)
    a, b, c= generator(4)
    print(a.shape)
    print(b.shape)
    print(c)
    print(generator.channels)