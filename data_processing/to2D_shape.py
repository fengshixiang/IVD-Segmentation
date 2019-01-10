from __future__ import print_function, division
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage.measurements as mear

#a = ['01', '02', '03', '04', '05', '06', '07', '08',\
#     '09', '10', '11', '12', '13', '14', '15', '16']
a = ['01', '06', '11']

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

for index in a:
    npy_data_address = '/DATA5_DB8/data/sxfeng/data/IVDM3Seg/npy_data/{}/{}_Labels.npy'.format(index, index)
    #data_save_address = '/DATA5_DB8/data/sxfeng/data/IVDM3Seg/2D_data/2D_data_all_shape/{}'.format(index) 
    data_save_address = '/DATA/data/sxfeng/Program/ensemble/greater/2D_data/{}'.format(index) 
    voxel = np.load(npy_data_address)
    voxel = del_small_rigion(voxel)

    labeled_voxel, _ = mear.label(voxel)
    smallest_arr = _select_small_IVD(labeled_voxel)
    
    print(len(smallest_arr[0]))
    point_list = np.where(voxel==1)
    x_min = min(point_list[1])
    x_max = max(point_list[1])
    y_min = min(point_list[2])
    y_max = max(point_list[2])
    small_x_min = min(smallest_arr[1])
    small_x_max = max(smallest_arr[1])
    small_y_min = min(smallest_arr[2])
    small_y_max = max(smallest_arr[2])
    x_min = x_min-3 if x_min-3>0 else 0
    x_max = x_max+3 if x_max+3<255 else 255
    y_min = y_min-3 if y_min-3>0 else 0
    y_max = y_max+3 if y_max+3<255 else 255
    arr = np.array([x_min, x_max, y_min, y_max, small_x_min, small_x_max, small_y_min, small_y_max])
    for addr in os.listdir(data_save_address):  # addr:0
        tmp_addr = os.path.join(data_save_address, addr)  #2D_data_all_shape/01/0
        for tmp in os.listdir(tmp_addr):
            if 'label.npy' in tmp:
                tmp_name = os.path.join(tmp_addr, tmp).replace('label.npy', 'range.txt')
        np.savetxt(tmp_name, arr)