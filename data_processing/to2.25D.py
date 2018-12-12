from __future__ import print_function, division
import numpy as np
import os
import matplotlib.pyplot as plt

npy_data_address = '/DATA5_DB8/data/sxfeng/data/IVDM3Seg/npy_data' 
data_save_address = '/DATA5_DB8/data/sxfeng/data/IVDM3Seg/2D_data/2.25D_12_data' 
start_index = 0
end_index = 36

Modality = ['fat', 'inn', 'opp', 'wat']

for addr in os.listdir(npy_data_address):
    print('process on {}'.format(addr))

    patient_addr = os.path.join(npy_data_address, addr)  #npy_data/01
    patient_save_addr = os.path.join(data_save_address, addr)   #2.5D_data/all/01
    if not os.path.exists(patient_save_addr):
        os.mkdir(patient_save_addr)

    fat_voxel_name = addr + '_fat.npy'
    inn_voxel_name = addr + '_inn.npy'
    opp_voxel_name = addr + '_opp.npy'
    wat_voxel_name = addr + '_wat.npy'
    label_name = addr + '_Labels.npy'

    fat_voxel_addr = os.path.join(patient_addr, fat_voxel_name)   #npy_data/01/01_fat.npy
    inn_voxel_addr = os.path.join(patient_addr, inn_voxel_name)
    opp_voxel_addr = os.path.join(patient_addr, opp_voxel_name)
    wat_voxel_addr = os.path.join(patient_addr, wat_voxel_name)
    voxel_label_addr = os.path.join(patient_addr, label_name)

    fat_voxel_arr = np.load(fat_voxel_addr)
    inn_voxel_arr = np.load(inn_voxel_addr)
    opp_voxel_arr = np.load(opp_voxel_addr)
    wat_voxel_arr = np.load(wat_voxel_addr)
    voxel_label_arr = np.load(voxel_label_addr)

    for index in range(start_index, end_index):
        index_addr = os.path.join(patient_save_addr, '{}'.format(index)) #2.5D_data/all/01/0
        if not os.path.exists(index_addr):
            os.mkdir(index_addr)

        fat_image_arr_5 = []
        inn_image_arr_5 = []
        opp_image_arr_5 = []
        wat_image_arr_5 = []
        for _ in range(3):
            fat_image_arr_5.append(fat_voxel_arr[index])
            inn_image_arr_5.append(inn_voxel_arr[index])
            opp_image_arr_5.append(opp_voxel_arr[index])
            wat_image_arr_5.append(wat_voxel_arr[index])

        image_label_arr = voxel_label_arr[index]
        label_save_addr = os.path.join(index_addr, '{}_{}_label.png'.format(addr, index))
        plt.imshow(image_label_arr, cmap='Greys_r')
        plt.savefig(label_save_addr)

        fat_imagearr_save_addr = os.path.join(index_addr, '{}_{}_fat.npy'.format(addr, index))
        inn_imagearr_save_addr = os.path.join(index_addr, '{}_{}_inn.npy'.format(addr, index))
        opp_imagearr_save_addr = os.path.join(index_addr, '{}_{}_opp.npy'.format(addr, index))
        wat_imagearr_save_addr = os.path.join(index_addr, '{}_{}_wat.npy'.format(addr, index))
        labelarr_save_addr = os.path.join(index_addr, '{}_{}_label.npy'.format(addr, index))

        np.save(fat_imagearr_save_addr, fat_image_arr_5)
        np.save(inn_imagearr_save_addr, inn_image_arr_5)
        np.save(opp_imagearr_save_addr, opp_image_arr_5)
        np.save(wat_imagearr_save_addr, wat_image_arr_5)
        np.save(labelarr_save_addr, image_label_arr)
        