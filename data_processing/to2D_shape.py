from __future__ import print_function, division
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage.measurements as mear

a = ['01', '02', '03', '04', '05', '06', '07', '08',\
     '09', '10', '11', '12', '13', '14', '15', '16']
#a = ['01', '06', '11']

def del_small_rigion(voxel):
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

def _select_small_IVD(voxel):
    nz = voxel.shape[0]
    nx = voxel.shape[1]
    ny = voxel.shape[2]
    gt_IVDset_list = []
    for i in range(7):
        gt_IVDset_list.append(set())
    for i in range(1, 8):
        tmp_arr = np.where(voxel==i)
        for j in range(len(tmp_arr[0])):
            gt_IVDset_list[i-1].add(tmp_arr[0][j]*nx*ny+tmp_arr[1][j]*ny+tmp_arr[2][j])

    tmp_len = 100000
    tmp_arr = 0
    for i in range(0,7):
        if len(gt_IVDset_list[i]) < tmp_len:
            tmp_len = len(gt_IVDset_list[i])
            tmp_arr = np.where(voxel==(i+1))
    return tmp_arr

def _select_small_IVD_2(voxel):
    gt_IVDlen_dict = dict()
    for i in range(1, 8):
        tmp_arr = np.where(voxel==i)
        gt_IVDlen_dict[i] = len(tmp_arr[0])

    sorted_list = sorted(gt_IVDlen_dict.items(), key= lambda x:x[1])
    small_1 = sorted_list[0][0]
    small_2 = sorted_list[1][0]
    tmp_arr_1 = np.where(voxel==small_1)
    tmp_arr_2 = np.where(voxel==small_2)
    x_min = min(min(tmp_arr_1[1]), min(tmp_arr_2[1]))
    x_max = max(max(tmp_arr_1[1]), max(tmp_arr_2[1]))
    y_min = min(min(tmp_arr_1[2]), min(tmp_arr_2[2]))
    y_max = max(max(tmp_arr_1[2]), max(tmp_arr_2[2]))
    return np.array([x_min, x_max, y_min, y_max])

def _region_of_IVD(voxel):
    gt_IVDy_dict = dict()
    for i in range(1, 8):
        tmp_arr = np.where(voxel==i)
        y_min = min(tmp_arr[2])
        gt_IVDy_dict[i] = y_min
    sorted_list = sorted(gt_IVDy_dict.items(), key=lambda x:x[1])
    list = []
    for i in range(7):
        index = sorted_list[i][0]
        tmp_arr = np.where(voxel==index)
        x_min = min(tmp_arr[1])
        x_max = max(tmp_arr[1])
        y_min = min(tmp_arr[2])
        y_max = max(tmp_arr[2])
        list.append([x_min, x_max, y_min, y_max])
    return list


for index in a:
    npy_data_address = '/DATA5_DB8/data/sxfeng/data/IVDM3Seg/npy_data_pianyi_left/{}/{}_Labels.npy'.format(index, index)
    data_save_address = '/DATA5_DB8/data/sxfeng/data/IVDM3Seg/2D_data/pianyi_left/{}'.format(index) 
    voxel = np.load(npy_data_address)
    voxel = del_small_rigion(voxel)

    labeled_voxel, _ = mear.label(voxel)
    
    point_list = np.where(voxel==1)
    x_min = min(point_list[1])
    x_max = max(point_list[1])
    y_min = min(point_list[2])
    y_max = max(point_list[2])

    point_list = _select_small_IVD(labeled_voxel)
    small_x_min = min(point_list[1])
    small_x_max = max(point_list[1])
    small_y_min = min(point_list[2])
    small_y_max = max(point_list[2])

    '''
    x_min = x_min-3 if x_min-3>0 else 1
    x_max = x_max+3 if x_max+3<255 else 255
    y_min = y_min-3 if y_min-3>0 else 1
    y_max = y_max+3 if y_max+3<255 else 255
    '''
    arr = np.array([x_min, x_max, y_min, y_max, small_x_min, small_x_max, small_y_min, small_y_max])
    for addr in os.listdir(data_save_address):  # addr:0
        tmp_addr = os.path.join(data_save_address, addr)  #2D_data_all_shape/01/0
        for tmp in os.listdir(tmp_addr):
            if 'label.npy' in tmp:
                tmp_name = os.path.join(tmp_addr, tmp).replace('label.npy', 'range.txt')
        np.savetxt(tmp_name, arr)