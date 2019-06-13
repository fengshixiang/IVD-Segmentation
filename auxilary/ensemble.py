from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import scipy.ndimage.measurements as mear
from Dice import Dice_3D, MDOC, SDDOC



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


save_address = '/DATA/data/sxfeng/Program/ensemble/voxel' 
gt_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/8modality_1/data/test_npydata'
root_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/ensemble_1'
root_address_list = []
root_address_1 = os.path.join(root_address, '5layer')
root_address_list.append(root_address_1)
root_address_2 = os.path.join(root_address, '4layer')
root_address_list.append(root_address_2)
root_address_3 = os.path.join(root_address, '3layer')
root_address_list.append(root_address_3)


############################################################
for i  in range(3):
    root_address_list[i] = os.path.join(root_address_list[i], 'result/pre_voxel')

dice_list = []
for index in os.listdir(root_address_list[0]):   # 01
    voxel = np.zeros([36, 256, 256])
    lmda = [0, 1, 0]
    for i in range(3):
        patient_addr = os.path.join(root_address_list[i], index) # result/pre_voxel/01
        patient_addr = os.path.join(patient_addr, 'pre.npy') # result/pre_voxel/01/pre.npy
        voxel += ((np.load(patient_addr) >= 1) * lmda[i])

    preVoxel = (voxel >= 1).astype(int)
    save_addr = os.path.join(save_address, index)
    if not os.path.exists(save_addr):
        os.mkdir(save_addr)

    gt_addr = os.path.join(gt_address, index)
    gt_addr = gt_addr + '/{}_Labels.npy'.format(index)
    gtVoxel = np.load(gt_addr)
    gtVoxel, _ = _del_small_rigion_gt(gtVoxel)
    labeledVoxel, labeled_num = _del_small_rigion_pre(preVoxel)
    print(labeled_num)
    np.save(save_addr+'/pre.npy', labeledVoxel)  # save labeled voxel
    gt_IVDset_list, pre_IVDset_list = _get_set(gtVoxel, labeledVoxel, labeled_num)
    image_dice_list = Dice_3D(gt_IVDset_list, pre_IVDset_list)
    dice_list.append(image_dice_list)
    print(image_dice_list)
mdoc = MDOC(dice_list)
sddoc = SDDOC(dice_list, mdoc)
print(mdoc, sddoc)


'''

############################################################
# ensemble crf voxel and to2D

for i  in range(3):
    root_address_list[i] = os.path.join(root_address_list[i], 'result/pre_voxel_crf')

for index in os.listdir(root_address_list[0]):   # 01
    voxel = np.zeros([36, 256, 256])
    lmda = [1, 1, 1]
    for i in range(3):
        patient_addr = os.path.join(root_address_list[i], index) # result/pre_voxel_crf/01
        patient_addr = os.path.join(patient_addr, 'pre.npy') # result/pre_voxel_crf/01/pre.npy
        voxel += (np.load(patient_addr) * lmda[i])

    preVoxel = voxel / 3.0
    save_addr = os.path.join(save_address, index)
    if not os.path.exists(save_addr):
        os.mkdir(save_addr)
    print(save_addr)

    np.save(save_addr+'/pre.npy', preVoxel)  # save labeled voxel


root_address = '/DATA/data/sxfeng/Program/ensemble/voxel'
save_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/010611_2/result/ensemble_for_crf'

if not os.path.exists(save_address):
    os.mkdir(save_address)

for index in os.listdir(root_address):
    index_addr = os.path.join(root_address, index)   #voxel/01
    index_save_addr = os.path.join(save_address, index)  #ensemble_for_crf/01
    index_addr = os.path.join(index_addr, 'pre.npy')  #voxel/01/pre.npy

    if not os.path.exists(index_save_addr):
        os.mkdir(index_save_addr)

    voxel = np.load(index_addr)
    for i in range(36):
        img_save_addr = os.path.join(index_save_addr, '{}'.format(i))   #ensemble_for_crf/01/0
        if not os.path.exists(img_save_addr):
            os.mkdir(img_save_addr)
        img_save_addr = os.path.join(img_save_addr, '{}.npy'.format(i))  #ensemble_for_crf/01/0/0.npy

        img = voxel[i]
        np.save(img_save_addr, img)

'''

