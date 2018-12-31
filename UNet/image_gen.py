# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Toy example, generates images at random that can be used for training

Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import glob
import os
from image_util import BaseDataProvider
from parameter import Parameter

class GrayScaleDataProvider(BaseDataProvider):
    channels = 1
    n_class = 2
    
    def __init__(self, nx, ny, **kwargs):
        super(GrayScaleDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3
        
    def _next_data(self):
        return create_image_and_label(self.nx, self.ny, **self.kwargs)

class RgbDataProvider(BaseDataProvider):
    channels = 3
    n_class = 2
    
    def __init__(self, nx, ny, **kwargs):
        super(RgbDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3

        
    def _next_data(self):
        data, label = create_image_and_label(self.nx, self.ny, **self.kwargs)
        return to_rgb(data), label

def create_image_and_label(nx,ny, cnt = 10, r_min = 5, r_max = 50, border = 92, sigma = 20, rectangles=False):
    
    
    image = np.ones((nx, ny, 1))
    label = np.zeros((nx, ny, 3), dtype=np.bool)
    mask = np.zeros((nx, ny), dtype=np.bool)
    for _ in range(cnt):
        a = np.random.randint(border, nx-border)
        b = np.random.randint(border, ny-border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1,255)

        y,x = np.ogrid[-a:nx-a, -b:ny-b]
        m = x*x + y*y <= r*r
        mask = np.logical_or(mask, m)

        image[m] = h

    label[mask, 1] = 1
    
    if rectangles:
        mask = np.zeros((nx, ny), dtype=np.bool)
        for _ in range(cnt//2):
            a = np.random.randint(nx)
            b = np.random.randint(ny)
            r =  np.random.randint(r_min, r_max)
            h = np.random.randint(1,255)
    
            m = np.zeros((nx, ny), dtype=np.bool)
            m[a:a+r, b:b+r] = True
            mask = np.logical_or(mask, m)
            image[m] = h
            
        label[mask, 2] = 1
        
        label[..., 0] = ~(np.logical_or(label[...,1], label[...,2]))
    
    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)
    
    if rectangles:
        return image, label
    else:
        return image, label[..., 1]

def to_rgb(img):
    img = img.reshape(img.shape[0], img.shape[1])
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    blue = np.clip(4*(0.75-img), 0, 1)
    red  = np.clip(4*(img-0.25), 0, 1)
    green= np.clip(44*np.fabs(img-0.5)-1., 0, 1)
    rgb = np.stack((red, green, blue), axis=2)
    return rgb

para = Parameter()
'''
class fourChannelProvider(BaseDataProvider):
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix="fat.npy",
                 mask_suffix='label.npy', shuffle_data=True, n_class = 2):
        super(fourChannelProvider, self).__init__(a_min, a_max)
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
'''
'''
class fourChannelProvider(BaseDataProvider):
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix="fat.npy",
                 mask_suffix='label.npy', shuffle_data=True, n_class = 2):
        super(fourChannelProvider, self).__init__(a_min, a_max)
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
        fat_path = path.replace(self.data_suffix, "opp.npy")
        inn_path = path.replace(self.data_suffix, "opp.npy")
        wat_path = path.replace(self.data_suffix, "opp.npy")
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
'''
class shapeProvider(BaseDataProvider):
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix="fat.npy",
                 mask_suffix='label.npy', shuffle_data=True, n_class = 2):
        super(shapeProvider, self).__init__(a_min, a_max)
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

    def _load_range(self, path, dtype=np.int32):
        return np.array(np.loadtxt(path), dtype=dtype)

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
        range_name = image_name.replace(self.data_suffix, 'range.txt')
        
        img = self._load_file(image_name, np.float32)
        label = self._load_label(label_name, np.bool)
        range_arr = self._load_range(range_name, np.int32)
    
        return img,label,range_arr

    def _load_data_and_label(self):
        data, label, range_arr = self._next_data()

        train_data = self._process_data(data)
        labels = self._process_labels(label)

        train_data, labels = self._post_process(train_data, labels)

        nx = train_data.shape[0]
        ny = train_data.shape[1]

        return train_data.reshape(1, nx, ny, self.channels), labels.reshape(1, ny, nx, self.n_class), range_arr

    def __call__(self, n):
        train_data, labels, range_arr = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
        Z = np.zeros((n, 4))

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
    root_address = para.root_address
    generator_address = os.path.join(root_address, 'data/train/*/*/*.npy')
    generator = shapeProvider(generator_address)
    a, b, c= generator(4)
    print(a.shape)
    print(b.shape)
    print(c)
    print(generator.channels)