from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import scipy.ndimage.measurements as mear
from Dice import Dice_3D, MDOC, SDDOC

import sys
sys.path.append(sys.path[0]+'/..')
from UNet.parameter import Parameter



para = Parameter()
root_address = para.root_address
voxel_address = os.path.join(root_address, 'data/test_npydata')
label_pre_address = os.path.join(root_address, 'result/pre_image')
voxel_save_address =os.path.join(root_address, 'result/pre_voxel')
labeled_voxel_save_address = os.path.join(root_address, 'result/labeled_voxel')


def img_resize_1(m):
    # 512 to 256
    new_m = np.zeros([256, 256])
    for i in range(256):
        for j in range(256):
            num = 0
            if m[2*i, 2*j] == True:
                num += 1
            if m[2*i+1, 2*j] == True:
                num += 1
            if m[2*i, 2*j+1] == True:
                num += 1
            if m[2*i+1, 2*j+1] == True:
                num += 1

            if num >=1:
                new_m[i, j] = 1
            else:
                new_m[i, j] = 0
    return new_m


def img_resize_2(m):
    # 128 to 256
    new_m = np.zeros([256, 256])
    for i in range(128):
        for j in range(128):
            new_m[2*i, 2*j] = new_m[2*i+1, 2*j] = new_m[2*i, 2*j+1] = new_m[2*i+1, 2*j+1] = m[i, j]

    return new_m.astype(int)


def img2voxel(label_pre_address, voxel_save_address, img_height=256, img_width=256):
    if not os.path.exists(voxel_save_address):
        os.mkdir(voxel_save_address)
    for addr in os.listdir(label_pre_address):   # addr: 01
        labels_addr = os.path.join(label_pre_address, addr)  # pre_image/01
        voxel_save_addr = os.path.join(voxel_save_address, addr)  # pre_voxel/01
        if not os.path.exists(voxel_save_addr):
            os.mkdir(voxel_save_addr)

        length_1 = len(os.listdir(labels_addr))  # length_1 = 36
        voxel = np.empty((length_1, img_width, img_height))

        for addr_1 in os.listdir(labels_addr):   #addr_1: 0
            label_addr = os.path.join(labels_addr, addr_1)    # pre_image/01/0
            for file in os.listdir(label_addr):
                if '_pre.npy' in file:
                    label_arr = np.load(os.path.join(label_addr, file))
                    if label_arr.shape[0] == 512:
                        label_arr = img_resize_1(label_arr)
                    elif label_arr.shape[0] == 128:
                        label_arr = img_resize_2(label_arr)
                    index = int(addr_1)
                    voxel[index] = label_arr

        np.save(voxel_save_addr + '/pre.npy', voxel)   ## pre_voxel/01/pre.npy


def img2voxel_xz(label_pre_address, voxel_save_address, img_height=36, img_width=256):
    if not os.path.exists(voxel_save_address):
        os.mkdir(voxel_save_address)
    for addr in os.listdir(label_pre_address):   # addr: 01
        labels_addr = os.path.join(label_pre_address, addr)  # pre_image/01
        voxel_save_addr = os.path.join(voxel_save_address, addr)  # pre_voxel/01
        if not os.path.exists(voxel_save_addr):
            os.mkdir(voxel_save_addr)

        length_1 = len(os.listdir(labels_addr))  
        voxel = np.empty((36, 256, 256))

        for addr_1 in os.listdir(labels_addr):   #addr_1: 0
            label_addr = os.path.join(labels_addr, addr_1)    # pre_image/01/0
            for file in os.listdir(label_addr):
                if '_pre.npy' in file:
                    label_arr = np.load(os.path.join(label_addr, file))
                    index = int(addr_1)
                    voxel[:, index, :] = label_arr

        np.save(voxel_save_addr + '/pre.npy', voxel)   ## pre_voxel/01/pre.npy


def img2voxel_yz(label_pre_address, voxel_save_address, img_height=36, img_width=256):
    if not os.path.exists(voxel_save_address):
        os.mkdir(voxel_save_address)
    for addr in os.listdir(label_pre_address):   # addr: 01
        labels_addr = os.path.join(label_pre_address, addr)  # pre_image/01
        voxel_save_addr = os.path.join(voxel_save_address, addr)  # pre_voxel/01
        if not os.path.exists(voxel_save_addr):
            os.mkdir(voxel_save_addr)

        length_1 = len(os.listdir(labels_addr))  
        voxel = np.empty((36, 256, 256))

        for addr_1 in os.listdir(labels_addr):   #addr_1: 0
            label_addr = os.path.join(labels_addr, addr_1)    # pre_image/01/0
            for file in os.listdir(label_addr):
                if '_pre.npy' in file:
                    label_arr = np.load(os.path.join(label_addr, file))
                    index = int(addr_1)
                    voxel[:, :, index] = label_arr

        np.save(voxel_save_addr + '/pre.npy', voxel)   ## pre_voxel/01/pre.npy


def _del_small_rigion_gt(voxel):
    #for gt voxel, find the top 7 rigions and delete others
    voxel, num = mear.label(voxel)
    if num==7:
        return voxel, num

    num_per_rigion = dict()
    for i in range(1, num+1):
        tmp_arr = np.where(voxel==i)
        num_per_rigion[i] = len(tmp_arr[0])

    d = sorted(num_per_rigion.items(), key= lambda x:x[1], reverse=True)
    index_lsit = []
    for i in range(7):
        index_lsit.append(d[i][0])
    for i in range(1, num+1):
        if not i in index_lsit:  # small region
            tmp_arr = np.where(voxel==i)
            for i in range(len(tmp_arr[0])):
                voxel[tmp_arr[0][i], tmp_arr[1][i], tmp_arr[2][i]] = 0

    # label the voxel again after detele small region
    voxel, num = mear.label(voxel)
    return voxel, num


def _del_small_rigion_pre(voxel):
    #for pre voxel, find the top top_num rigions and delete others
    voxel, num = mear.label(voxel)  
    num_per_rigion = dict()
    for i in range(1, num+1):
        tmp_arr = np.where(voxel==i)
        if len(tmp_arr[0]) > 500:
            for j in range(len(tmp_arr[0])):
                voxel[tmp_arr[0][j], tmp_arr[1][j], tmp_arr[2][j]] = 1
            num_per_rigion[i] = len(tmp_arr[0])
        else:
            for j in range(len(tmp_arr[0])):
                voxel[tmp_arr[0][j], tmp_arr[1][j], tmp_arr[2][j]] = 0

    # label the voxel again after detele small region
    voxel, num = mear.label(voxel)
    return voxel, num


def _get_set(gtVoxel, preVoxel, labeled_num):
    """
    Get the list of gtVoxel set and list of preVoxel set, and match them

    :param gtVoxel: labeled gtVoxel
    :param preVoxel: labeled preVoxel
    """
    nz = gtVoxel.shape[0]
    nx = gtVoxel.shape[1]
    ny = gtVoxel.shape[2]
    gt_IVDset_list = []
    pre_IVDset_list = []
    for i in range(7):
        gt_IVDset_list.append(set())
    for i in range(labeled_num):
        pre_IVDset_list.append(set())

    for i in range(1, 8):
        tmp_arr = np.where(gtVoxel==i)
        for j in range(len(tmp_arr[0])):
            gt_IVDset_list[i-1].add(tmp_arr[0][j]*nx*ny+tmp_arr[1][j]*ny+tmp_arr[2][j])
    for i in range(1, labeled_num+1):
        tmp_arr = np.where(preVoxel==i)
        for j in range(len(tmp_arr[0])):
            pre_IVDset_list[i-1].add(tmp_arr[0][j]*nx*ny+tmp_arr[1][j]*ny+tmp_arr[2][j])

    # match
    for i in range(7):
        tmp_set = gt_IVDset_list[i]
        for j in range(i, labeled_num):
            if len(tmp_set & pre_IVDset_list[j]) > 50:
                tmp_set_1 = pre_IVDset_list[i]
                pre_IVDset_list[i] = pre_IVDset_list[j]
                pre_IVDset_list[j] = tmp_set_1

    print([len(gt_IVDset_list[i]) for i in range(7)])
    print([len(pre_IVDset_list[i]) for i in range(7)])
    return gt_IVDset_list, pre_IVDset_list[0:7]


def eval(gt_voxel_address, pre_voxel_address, labeled_voxel_save_address):
    
    if not os.path.exists(labeled_voxel_save_address):
        os.mkdir(labeled_voxel_save_address)

    gt_voxel_addr = os.listdir(gt_voxel_address)
    gt_voxel_addr.sort()
    gt_voxel_addr = [os.path.join(gt_voxel_address, i) for i in gt_voxel_addr] # test_npydata/01
    pre_voxel_addr = os.listdir(pre_voxel_address)
    pre_voxel_addr.sort()
    pre_voxel_addr = [os.path.join(pre_voxel_address, i) for i in pre_voxel_addr] # pre_voxel/01
    labeled_voxel_save_addr = [path.replace('pre_voxel', 'labeled_voxel') for path in pre_voxel_addr]  # labeled_voxel/01

    dice_list = []
    for i in range(len(gt_voxel_addr)):
        gt_addr = ''
        for filename in os.listdir(gt_voxel_addr[i]): 
            if 'Labels.npy' in filename:
                gt_addr = os.path.join(gt_voxel_addr[i], filename)  # test_npydata/01/01_Labels.npy

        pre_addr = os.path.join(pre_voxel_addr[i], 'pre.npy')  # pre_voxel/01/pre.npy
        print('evaluation on {}'.format(pre_addr))
        
        if not os.path.exists(labeled_voxel_save_addr[i]):
            os.mkdir(labeled_voxel_save_addr[i])
        labeled_addr = os.path.join(labeled_voxel_save_addr[i], 'labeled.npy')

        gtVoxel = np.load(gt_addr)
        preVoxel = np.load(pre_addr)
        gtVoxel, _ = _del_small_rigion_gt(gtVoxel)
        labeledVoxel, labeled_num = _del_small_rigion_pre(preVoxel)
        print(labeled_num)
        np.save(labeled_addr, labeledVoxel)  # save labeled voxel
        gt_IVDset_list, pre_IVDset_list = _get_set(gtVoxel, labeledVoxel, labeled_num)
        image_dice_list = Dice_3D(gt_IVDset_list, pre_IVDset_list)
        dice_list.append(image_dice_list)
        print(image_dice_list)

    mdoc = MDOC(dice_list)
    sddoc = SDDOC(dice_list, mdoc)
    return mdoc, sddoc


if __name__ == '__main__':
    img2voxel(label_pre_address, voxel_save_address)
    print('start evaluation')
    mdoc, sddoc = eval(voxel_address, voxel_save_address,  labeled_voxel_save_address)
    print(mdoc)
    print(sddoc)


