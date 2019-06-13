from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage.measurements as mear
import nibabel as nib

data_address = '/DATA/data/sxfeng/data/IVDM3Seg/Training' 
save_address = '/DATA/data/sxfeng/Program/ensemble/histogram'
'''
for adr in os.listdir(data_address):
    patient_adr = os.path.join(data_address, adr)    # Training/01
    save_adr = os.path.join(save_address, adr)     #histogram/01
    if not os.path.exists(save_adr):
        os.mkdir(save_adr)

    for adr_1 in os.listdir(patient_adr): 
        image_adr = os.path.join(patient_adr, adr_1)  # Training/01/01_fat.nii
        print(image_adr)
        img = nib.load(image_adr).get_data()
        img_arr = np.asarray(img)
        img_arr = img_arr.reshape([-1])
        img_list = img_arr.tolist()
        plt.hist(img_list, bins=100, density=True, facecolor='g')
        #plt.axis([0, 500, 0, 0.075])
        save_addr = os.path.join(save_adr, adr_1)
        save_addr = save_addr.replace('.nii', '.png')
        plt.savefig(save_addr)

'''

a = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']
for tmp_a in a:
    img = nib.load('/DATA/data/sxfeng/data/IVDM3Seg/Training/{}/{}_fat.nii'.format(tmp_a, tmp_a)).get_data()
    img_arr = np.asarray(img)
    #img_arr = img_arr[18].reshape([-1])
    img_arr = img_arr.reshape([-1])
    img_list = img_arr.tolist()
    plt.hist(img_list, bins=100, density=True, facecolor='g')
    plt.axis([0, 500, 0, 0.07])
    plt.savefig('/DATA/data/sxfeng/Program/ensemble/histogram/{}_fat.png'.format(tmp_a))


