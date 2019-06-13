from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage.measurements as mear
from Dice import Dice_3D, MDOC, SDDOC
from skimage import measure

#label_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/010611_1/data/test_npydata'
#pre_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/512-010611-IVD/result/pre_voxel'
label_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/ensemble_IVD/create6_EPE3/5layer/result/pre_voxel'
pre_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/ensemble_IVD/create6_EPE3/4layer/result/pre_voxel'
save_address = '/DATA/data/sxfeng/Program/ensemble/view'
if not os.path.exists(save_address):
    os.mkdir(save_address)

for index in os.listdir(pre_address):
    print(index)
    patient_pre_addr = os.path.join(pre_address, index)
    patient_pre_addr = os.path.join(patient_pre_addr, 'pre.npy')
    patient_label_addr = os.path.join(label_address, index)   # pre_voxel/01
    #patient_label_addr = os.path.join(patient_label_addr, '{}_Labels.npy'.format(index))
    patient_label_addr = os.path.join(patient_label_addr, 'pre.npy'.format(index))
    patient_save_addr = os.path.join(save_address, index)   # view/01
    if not os.path.exists(patient_save_addr):
        os.mkdir(patient_save_addr)

    label_voxel = np.load(patient_label_addr)
    pre_voxel = np.load(patient_pre_addr)
    for i in range(36):
        label_img = label_voxel[i]
        pre_img = pre_voxel[i]
        img = np.zeros([256, 256, 3])
        img[:, :, :] = 100
        label_arr = np.where(label_img==1)
        pre_arr = np.where(pre_img==1)


        label_len = len(label_arr[0])
        pre_len = len(pre_arr[0])
        
        if not label_len==0:
            for l in range(label_len):
                #img[label_arr[0][l], label_arr[1][l], 0] = 0
                img[label_arr[0][l], label_arr[1][l], 1] = 0
        '''
        contours = measure.find_contours(pre_img, 0.5)
        for tmp in contours:
            for [x, y] in tmp:
                #img[int(x), int(y), 0] = 0
                img[int(x), int(y), 2] = 0

        '''
        if not pre_len==0:
            for l in range(pre_len):
                #img[label_arr[0][l], label_arr[1][l], 0] = 0
                img[pre_arr[0][l], pre_arr[1][l], 2] = 0
                img[pre_arr[0][l], pre_arr[1][l], 0] = 0

        save_addr = os.path.join(patient_save_addr, '{}.png'.format(i))     #view/01/0.png
        plt.imshow(img)
        plt.savefig(save_addr)




