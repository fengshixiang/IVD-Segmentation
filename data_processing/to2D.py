from __future__ import print_function, division
import numpy as np
import os
import matplotlib.pyplot as plt

npy_data_address = '/DATA5_DB8/data/sxfeng/data/IVDM3Seg/npy_data' 
data_save_address = '/DATA5_DB8/data/sxfeng/data/IVDM3Seg/2D_data/2D_data_inn' 
start_index = 0
end_index = 36


for addr in os.listdir(npy_data_address):
	print('process on {}'.format(addr))

	patient_addr = os.path.join(npy_data_address, addr)
	if not os.path.exists(patient_addr):
		os.mkdir(patient_addr)
	patient_save_addr = os.path.join(data_save_address, addr)
	if not os.path.exists(patient_save_addr):
		os.mkdir(patient_save_addr)

	#fat_voxel_name = addr + '_fat.npy'
	inn_voxel_name = addr + '_inn.npy'
	#opp_voxel_name = addr + '_opp.npy'
	#wat_voxel_name = addr + '_wat.npy'
	label_name = addr + '_Labels.npy'

	#fat_voxel_addr = os.path.join(patient_addr, fat_voxel_name)
	inn_voxel_addr = os.path.join(patient_addr, inn_voxel_name)
	#opp_voxel_addr = os.path.join(patient_addr, opp_voxel_name)
	#wat_voxel_addr = os.path.join(patient_addr, wat_voxel_name)
	voxel_label_addr = os.path.join(patient_addr, label_name)

	#fat_voxel_arr = np.load(fat_voxel_addr)
	inn_voxel_arr = np.load(inn_voxel_addr)
	#opp_voxel_arr = np.load(opp_voxel_addr)
	#wat_voxel_arr = np.load(wat_voxel_addr)
	voxel_label_arr = np.load(voxel_label_addr)

	for index in range(start_index, end_index):
		index_addr = os.path.join(patient_save_addr, '{}'.format(index))
		if not os.path.exists(index_addr):
			os.mkdir(index_addr)

		#fat_image_arr = fat_voxel_arr[index]
		inn_image_arr = inn_voxel_arr[index]
		#opp_image_arr = opp_voxel_arr[index]
		#wat_image_arr = wat_voxel_arr[index]
		image_label_arr = voxel_label_arr[index]

		#fat_image_save_addr = os.path.join(index_addr, '{}_{}_fat.png'.format(addr, index))
		inn_image_save_addr = os.path.join(index_addr, '{}_{}_inn.png'.format(addr, index))
		#opp_image_save_addr = os.path.join(index_addr, '{}_{}_opp.png'.format(addr, index))
		#wat_image_save_addr = os.path.join(index_addr, '{}_{}_wat.png'.format(addr, index))
		label_save_addr = os.path.join(index_addr, '{}_{}_label.png'.format(addr, index))

		#fat_imagearr_save_addr = os.path.join(index_addr, '{}_{}_fat.npy'.format(addr, index))
		inn_imagearr_save_addr = os.path.join(index_addr, '{}_{}_inn.npy'.format(addr, index))
		#opp_imagearr_save_addr = os.path.join(index_addr, '{}_{}_opp.npy'.format(addr, index))
		#wat_imagearr_save_addr = os.path.join(index_addr, '{}_{}_wat.npy'.format(addr, index))
		labelarr_save_addr = os.path.join(index_addr, '{}_{}_label.npy'.format(addr, index))
		
		#plt.imshow(fat_image_arr, cmap='Greys_r')
		#plt.savefig(fat_image_save_addr)
		plt.imshow(inn_image_arr, cmap='Greys_r')
		plt.savefig(inn_image_save_addr)
		#plt.imshow(opp_image_arr, cmap='Greys_r')
		#plt.savefig(opp_image_save_addr)
		#plt.imshow(wat_image_arr, cmap='Greys_r')
		#plt.savefig(wat_image_save_addr)
		plt.imshow(image_label_arr, cmap='Greys_r')
		plt.savefig(label_save_addr)

		#np.save(fat_imagearr_save_addr, fat_image_arr)
		np.save(inn_imagearr_save_addr, inn_image_arr)
		#np.save(opp_imagearr_save_addr, opp_image_arr)
		#np.save(wat_imagearr_save_addr, wat_image_arr)
		np.save(labelarr_save_addr, image_label_arr)
