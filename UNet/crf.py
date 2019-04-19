from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import sys
sys.path.append(sys.path[0]+'/..')
from UNet.parameter import Parameter
para = Parameter()
root_address = para.root_address
img_address = os.path.join(root_address, 'data/test')
pre_img_address = os.path.join(root_address, 'result/ensemble_for_crf')
img_save_address =os.path.join(root_address, 'result/pre_image_after_crf')
if not os.path.exists(img_save_address):
    os.mkdir(img_save_address)

def crf_inference_1(img, probs, t=10, scale_factor=1, labels=2):
    """
    img.shape:   h x w x 3
    probs.shape: h x w
    """
    h, w = img.shape[:2]
    n_labels = labels
    
    probs = np.expand_dims(probs, 0)
    probs = np.append(1 - probs, probs, axis=0)
    
    d = dcrf.DenseCRF2D(w, h, n_labels)

    U = -np.log(probs+0.00001)   #....
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U).astype(np.float32)
    img = np.ascontiguousarray(img)
    
    d.setUnaryEnergy(U)
    
    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=np.copy(img), compat=10)
    
    Q = d.inference(10)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q

def crf_inference_2(img, probs, t=10, scale_factor=1, labels=2):

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80 / scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    # d.addPairwiseGaussian(sxy=8, compat=3)
    # d.addPairwiseBilateral(sxy=15, srgb=5, rgbim=np.copy(img), compat=10)
    
    
    #     im = Image.fromarray(img)
    #     im = im.convert("L")
    #     im = np.asarray(im)
    #     pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=im, chdim=-1)
    #     d.addPairwiseEnergy(pairwise_energy, compat=10)

    Q = d.inference(t)
    #Q = np.array(Q).reshape((n_labels, h, w))
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q

def crf_inference_3(img, probs, t=10, scale_factor=1, labels=2):
    h, w = img.shape[:2]
    n_labels = labels
    U = unary_from_softmax(probs)  # note: num classes is first dim
    d = dcrf.DenseCRF2D(w, h, n_labels)
    d.setUnaryEnergy(U)
    pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.001,), img=img, chdim=2)
    d.addPairwiseEnergy(pairwise_energy, compat=10)
    Q_unary = d.inference(10)
    map_soln_unary = np.argmax(Q_unary, axis=0)
    map_soln_unary = map_soln_unary.reshape((h,w))
    return map_soln_unary


def read_img_3(img_path, pre_path):
    img = np.load(img_path)  #[256,256]
    img = img.astype(np.uint8)
    img = np.expand_dims(img, 2)

    probs = np.load(pre_path)
    probs = np.expand_dims(probs, 0)
    probs = np.append(1 - probs, probs, axis=0)  #[2, 256, 256]
    return img, probs

def read_img_4(img_path, pre_path):
    num = len(img_path)
    img = np.zeros([256, 256, num])
    for i in range(num):
        path = img_path[i]
        img[..., i] = np.load(path)  #[256,256]
    img = img.astype(np.uint8)
    probs = np.load(pre_path)
    probs = np.expand_dims(probs, 0)
    probs = np.append(1 - probs, probs, axis=0)  #[2, 256, 256]
    return img, probs


def crf():
    for index in os.listdir(img_address): #01
        print(index)
        index_addr = os.path.join(img_address, index)   #test/01
        index_pre_addr = os.path.join(pre_img_address, index)  #ensemble_for_crf/01
        index_save_addr = os.path.join(img_save_address, index)  #pre_img_after_crf/01

        if not os.path.exists(index_save_addr):
            os.mkdir(index_save_addr)

        for addr in os.listdir(index_addr):  #0
            img_addr = os.path.join(index_addr, addr)  #test/01/0
            img_pre_addr = os.path.join(index_pre_addr, addr)  #ensemble_for_crf/01/0
            img_save_addr = os.path.join(index_save_addr, addr)  #pre_img_after_crf/01/0

            if not os.path.exists(img_save_addr):
                os.mkdir(img_save_addr)

            img_path = os.path.join(img_addr, '{}_{}_opp.npy'.format(index, addr))
            img_pre_path = os.path.join(img_pre_addr, '{}.npy'.format(addr))
            img_save_path = os.path.join(img_save_addr, '{}_{}_fat_pre.npy'.format(index, addr))
            img, probs = read_img_3(img_path, img_pre_path)
            crf_score = crf_inference_3(img, probs)
            crf_score = crf_score > para.mask
            np.save(img_save_path, crf_score)
            plt.imshow(crf_score, cmap='gray')
            plt.savefig(img_save_path.replace('npy', 'png'))

def crf_2():
    for index in os.listdir(img_address): #01
        print(index)
        index_addr = os.path.join(img_address, index)   #test/01
        index_pre_addr = os.path.join(pre_img_address, index)  #ensemble_for_crf/01
        index_save_addr = os.path.join(img_save_address, index)  #pre_img_after_crf/01

        if not os.path.exists(index_save_addr):
            os.mkdir(index_save_addr)

        for addr in os.listdir(index_addr):  #0
            img_addr = os.path.join(index_addr, addr)  #test/01/0
            img_pre_addr = os.path.join(index_pre_addr, addr)  #ensemble_for_crf/01/0
            img_save_addr = os.path.join(index_save_addr, addr)  #pre_img_after_crf/01/0

            if not os.path.exists(img_save_addr):
                os.mkdir(img_save_addr)

            img_path_1 = os.path.join(img_addr, '{}_{}_fat.npy'.format(index, addr))
            img_path_2 = os.path.join(img_addr, '{}_{}_inn.npy'.format(index, addr))
            img_path_3 = os.path.join(img_addr, '{}_{}_opp.npy'.format(index, addr))
            img_path_4 = os.path.join(img_addr, '{}_{}_wat.npy'.format(index, addr))
            img_path = [img_path_1, img_path_2, img_path_3, img_path_4]
            img_pre_path = os.path.join(img_pre_addr, '{}.npy'.format(addr))
            img_save_path = os.path.join(img_save_addr, '{}_{}_fat_pre.npy'.format(index, addr))
            img, probs = read_img_4(img_path, img_pre_path)
            crf_score = crf_inference_3(img, probs)
            crf_score = crf_score > 0.1
            np.save(img_save_path, crf_score)
            plt.imshow(crf_score, cmap='gray')
            plt.savefig(img_save_path.replace('npy', 'png'))

if __name__ == '__main__':
    crf_2()


