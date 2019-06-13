from __future__ import print_function, division
import numpy as np
import nibabel as nib
import os

data_address = '/DATA/data/sxfeng/data/IVDM3Seg/Training' 
data_save_address = '/DATA5_DB8/data/sxfeng/data/IVDM3Seg/npy_data_label_img_8m' 

if not os.path.exists(data_save_address):
	os.mkdir(data_save_address)

for adr in os.listdir(data_address):
	patient_adr = os.path.join(data_address, adr)
	patient_save_adr = os.path.join(data_save_address, adr)
	if not os.path.exists(patient_save_adr):
		os.mkdir(patient_save_adr)

	for adr_1 in os.listdir(patient_adr):
		if 'Labels' in adr_1:
			img_adr = os.path.join(patient_adr, adr_1)
			img = nib.load(img_adr).get_data()
			img_arr = np.asarray(img)
			img_save_adr = os.path.join(patient_save_adr, adr_1[0:-3] + 'npy')
			np.save(img_save_adr, img_arr)
		else:
			img_adr = os.path.join(patient_adr, adr_1)
			img = nib.load(img_adr).get_data()
			img_mean = np.mean(img)
			img_std = np.std(img)
			img = (img - img_mean) / img_std
			img_arr = np.asarray(img)
			img_save_adr = os.path.join(patient_save_adr, adr_1[0:-3] + 'npy')
			np.save(img_save_adr, img_arr)

			label_adr = img_adr[0:-7] + 'Labels.nii'
			label = nib.load(label_adr).get_data()
			img_label = img + label
			img_label_mean = np.mean(img_label)
			img_label_std = np.std(img_label)
			img_label = (img_label - img_label_mean) / img_label_std
			img_label_arr = np.asarray(img_label)
			img_label_save_adr = img_save_adr.replace('.npy', 'l.npy')
			np.save(img_label_save_adr, img_arr)
