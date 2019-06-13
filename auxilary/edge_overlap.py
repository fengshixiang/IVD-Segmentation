from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage.measurements as mear
from Dice import Dice_3D, MDOC, SDDOC
from skimage import measure

label_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/8modality/data/test_npydata'
pre_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/8modality/result/pre_voxel'
save_address = '/DATA/data/sxfeng/Program/ensemble/edge'

def _find_contour(label_arr, contour_arr, a):
    s = set()
    nx = label_arr.shape[0]
    ny = label_arr.shape[1]
    for tmp in contour_arr:
        for x, y in tmp:
            x = int(x)
            y = int(y)
            if x+1 < nx:
                if label_arr[x+1, y] == a:
                    s.add((x+1, y))
            if x-1 >= 0:
                if label_arr[x-1, y] == a:
                    s.add((x-1, y))
            if y+1 < ny:
                if label_arr[x, y+1] == a:
                    s.add((x, y+1))
            if y-1 >= 0:
                if label_arr[x, y-1] == a:
                    s.add((x, y-1))
    #print(len(s))
    return s

def _del_small_rigion(voxel):
    #for gt voxel, find the top 7 rigions and delete others
    voxel, num = mear.label(voxel)

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
        elif i in index_lsit:
            tmp_arr = np.where(voxel==i)
            for i in range(len(tmp_arr[0])):
                voxel[tmp_arr[0][i], tmp_arr[1][i], tmp_arr[2][i]] = 1

    return voxel

def single_IVD(voxel):
    voxel, _ = mear.label(voxel)
    nz = voxel.shape[0]
    nx = voxel.shape[1]
    ny = voxel.shape[2]
    gt_IVD_ymin_dict = dict()
    gt_IVD_range_list = []
    for i in range(1, 8):
        tmp_arr = np.where(voxel==i)
        length = len(tmp_arr[0])
        x_min = min(tmp_arr[1])
        x_max = max(tmp_arr[1])
        y_min = min(tmp_arr[2])
        y_max = max(tmp_arr[2])
        gt_IVD_ymin_dict[i-1] = y_min
        range_arr = np.array([x_min-2, x_max+2, y_min-2, y_max+2, length])
        gt_IVD_range_list.append(range_arr)

    sorted_ymin = sorted(gt_IVD_ymin_dict.items(), key= lambda x:x[1])
    new_range_list = []
    for i in range(7):
        index = sorted_ymin[i][0]
        new_range_list.append(gt_IVD_range_list[index])
    return new_range_list

def _addIndex(contour_set, index):
    new_set = set()
    for point in contour_set:
        new_set.add((index, point[0], point[1]))
    return new_set


length_list = []
overlap_list = []
for index in os.listdir(pre_address):
    print(index)
    patient_pre_addr = os.path.join(pre_address, index)
    patient_pre_addr = os.path.join(patient_pre_addr, 'pre.npy')
    patient_label_addr = os.path.join(label_address, index)
    patient_label_addr = os.path.join(patient_label_addr, '{}_Labels.npy'.format(index))

    label_voxel = np.load(patient_label_addr)
    pre_voxel = np.load(patient_pre_addr)
    range_list = single_IVD(_del_small_rigion(label_voxel))

    tmp_list = []
    for i in range(7):
        single_range = range_list[i]
        single_label_voxel = label_voxel[:, single_range[0]:single_range[1], single_range[2]:single_range[3]]
        single_pre_voxel = pre_voxel[:, single_range[0]:single_range[1], single_range[2]:single_range[3]]
        voxel_label_set = set()
        voxel_pre_set = set()
        for j in range(36):
            label_img = single_label_voxel[j]
            pre_img = single_pre_voxel[j]
            label_con_set = _find_contour(label_img, measure.find_contours(label_img, 0), 255)
            pre_con_set = _find_contour(pre_img, measure.find_contours(pre_img, 0), 1)
            label_con_set = _addIndex(label_con_set, j)
            pre_con_set = _addIndex(pre_con_set, j)
            voxel_label_set = voxel_label_set|label_con_set
            voxel_pre_set = voxel_pre_set|pre_con_set
        intersection = voxel_pre_set&voxel_label_set
        tmp_list.append([single_range[4], len(intersection)/len(voxel_pre_set)])


    print(tmp_list)
    tmp_list_1 = [i[0] for i in tmp_list]
    tmp_list_2 = [i[1] for i in tmp_list]
    length_list.append(tmp_list_1)
    overlap_list.append(tmp_list_2)
    save_addr = os.path.join(save_address, '{}.png'.format(index))     #view/01/0.png
    #plt.figure()
    plt.scatter(tmp_list_1, tmp_list_2)
    plt.xlabel('Pixel num') 
    plt.ylabel('Edge Dice') 
    plt.savefig(save_addr)