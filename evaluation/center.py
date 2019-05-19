from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as ndimg

import sys
sys.path.append(sys.path[0]+'/..')
from UNet.parameter import Parameter



para = Parameter()
root_address = para.root_address
voxel_address = os.path.join(root_address, 'data/test_npydata')
label_pre_address = os.path.join(root_address, 'result/pre_image')
voxel_save_address =os.path.join(root_address, 'result/pre_voxel')
labeled_voxel_save_address = os.path.join(root_address, 'result/labeled_voxel')

def Dice_3D(gt_IVDset_list, pre_IVDset_list):
    length = len(gt_IVDset_list)
    image_list = []
    for i in range(length):
        gt_len = len(gt_IVDset_list[i])
        pre_len = len(pre_IVDset_list[i])
        union_len = len(gt_IVDset_list[i] & pre_IVDset_list[i])
        dice = 2*union_len/(gt_len + pre_len)
        image_list.append(dice)
    return image_list

def MDOC(dice_list):
    total = 0
    for li in dice_list:
        total += sum(li)

    mdoc = total/len(dice_list)/7
    return mdoc

def SDDOC(dice_list, mdoc):
    total = 0
    for li in dice_list:
        for dice in li:
            total += pow(dice-mdoc, 2)

    sddoc = total/(len(dice_list)*7-1)
    sddoc = np.sqrt(sddoc)
    return sddoc

def MDIST(dist_list):
    total = 0
    for li in dist_list:
        total += sum(li)
    return total/21

def SDDIST(dist_list, mdist):
    total = 0
    for li in dist_list:
        for dist in li:
            total += pow(dist - mdist, 2)
    sddist = total/(len(dist_list)*7-1)
    sddist = np.sqrt(sddist)
    return sddist

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
                    index = int(addr_1)
                    voxel[index] = label_arr

        np.save(voxel_save_addr + '/pre.npy', voxel)   ## pre_voxel/01/pre.npy

def _del_small_rigion_gt(voxel):
    #for gt voxel, find the top 7 rigions and delete others
    voxel, num = ndimg.label(voxel)
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
    voxel, num = ndimg.label(voxel)
    return voxel, num

def _del_small_rigion_pre(voxel):
    #for pre voxel, find the top top_num rigions and delete others
    voxel, num = ndimg.label(voxel)  
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
    voxel, num = ndimg.label(voxel)
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

def plt_histogram(gt_len_list, dice_list):
    save_addr = os.path.join(root_address, 'result/hist')
    if not os.path.exists(save_addr):
        os.mkdir(save_addr)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.scatter(gt_len_list[0], dice_list[0], marker='x')
    plt.subplot(2, 2, 2)
    plt.scatter(gt_len_list[1], dice_list[1], marker='+')
    plt.subplot(2, 2, 3)
    plt.scatter(gt_len_list[2], dice_list[2], marker='o')
    plt.subplot(2, 2, 4)
    plt.scatter(gt_len_list[0], dice_list[0], marker='x')
    plt.scatter(gt_len_list[1], dice_list[1], marker='+')
    plt.scatter(gt_len_list[2], dice_list[2], marker='o')
    plt.savefig(save_addr+'/result.png')

def distance(point1, point2):
    sum = 0
    sum += np.square(point1[0] - point2[0])*1.25**2
    sum += np.square(point1[1] - point2[1])*1.25**2
    sum += np.square(point1[2] - point2[2])*2.0**2

    return np.sqrt(sum)

def match_point(point_list1, point_list2):
    dist_list = []
    for i in range(7):
        tmp_point = point_list1[i]
        for j in range(7):
            if distance(tmp_point, point_list2[j]) < 4:
                dist_list.append(distance(tmp_point, point_list2[j]))

    return dist_list

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
    gt_len_list = []
    dist_list = []
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

        point_list1 = ndimg.center_of_mass(gtVoxel, gtVoxel, [1,2,3,4,5,6,7])
        point_list2 = ndimg.center_of_mass(labeledVoxel, labeledVoxel, [1,2,3,4,5,6,7])

        gt_IVDset_list, pre_IVDset_list = _get_set(gtVoxel, labeledVoxel, labeled_num)
        image_gt_len_list = [len(gt_IVDset_list[i]) for i in range(7)]
        image_dice_list = Dice_3D(gt_IVDset_list, pre_IVDset_list)
        gt_len_list.append(image_gt_len_list)
        dice_list.append(image_dice_list)
        dist_list.append(match_point(point_list1, point_list2))
        print(image_dice_list)

    mdoc = MDOC(dice_list)
    sddoc = SDDOC(dice_list, mdoc)
    mdist = MDIST(dist_list)
    sddist = SDDIST(dist_list, mdist)
    print(dist_list)
    plt_histogram(gt_len_list, dice_list)
    return mdoc, sddoc, mdist, sddist

if __name__ == "__main__":
    img2voxel(label_pre_address, voxel_save_address)
    print('start evaluation')
    mdoc, sddoc, mdist, sddist = eval(voxel_address, voxel_save_address,  labeled_voxel_save_address)
    print(mdoc, sddoc, mdist, sddist)