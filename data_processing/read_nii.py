from __future__ import print_function, division
import numpy as np
import nibabel as nib
import os

data_address = '/DATA/data/sxfeng/data/IVDM3Seg/Training' 
data_save_address = '/DATA5_DB8/data/sxfeng/data/IVDM3Seg/npy_data_norm' 

if not os.path.exists(data_save_address):
	os.mkdir(data_save_address)

for adr in os.listdir(data_address):
	patient_adr = os.path.join(data_address, adr)
	patient_save_adr = os.path.join(data_save_address, adr)
	if not os.path.exists(patient_save_adr):
		os.mkdir(patient_save_adr)

	for adr_1 in os.listdir(patient_adr):
		if 'Labels' in adr_1:
			image_adr = os.path.join(patient_adr, adr_1)
			img = nib.load(image_adr).get_data()
			#img_max = np.amax(img)
			#img = np.clip(img, 0, 255)
			img_arr = np.asarray(img)
			image_save_adr = os.path.join(patient_save_adr, adr_1[0:-3] + 'npy')
			np.save(image_save_adr, img_arr)
		else:
			image_adr = os.path.join(patient_adr, adr_1)
			img = nib.load(image_adr).get_data()
			#img_max = np.amax(img)
			#img = np.clip(img, 0, 255)
			img_mean = np.mean(img)
			img_std = np.std(img)
			img = (img - img_mean) / img_std
			img_arr = np.asarray(img)
			image_save_adr = os.path.join(patient_save_adr, adr_1[0:-3] + 'npy')
			np.save(image_save_adr, img_arr)