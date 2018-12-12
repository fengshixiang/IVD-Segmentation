'''
For evaluating the segmentation performance, 
calculate the Dice overlap coefficients(Dice), 
Average absolute distance(AAD) and Hausdorff distance(HD)
Dice measures the percentage of correctly segmented voxels
AAD measures the average absolute distance form the ground truth IVD surface and the predicted surface.
HD measures the Hausdorff distance between the ground truth IVD surface and predicted surface.

For evaluating the localization performance,
calculate the Mean localizatin distance(MLD) with standard deviation(SD)
and Successful detection rate(SDR)
'''

from __future__ import division, print_function
import numpy as np 


def Dice_3D(gt_IVDset_list, pre_IVDset_list):
	length = len(gt_IVDset_list)
	image_list = []
	for i in range(length):
		gt_len = len(gt_IVDset_list[i])
		pre_len = len(pre_IVDset_list[i])
		union_len = len(gt_IVDset_list[i] & pre_IVDset_list[i])
		dice = 2*union_len/(gt_len + pre_len)
		image_list.append(dice)
	return image_list


def MDOC(dice_list):
	total = 0
	for li in dice_list:
		total += sum(li)

	mdoc = total/len(dice_list)/7
	return mdoc


def SDDOC(dice_list, mdoc):
	total = 0
	for li in dice_list:
		for dice in li:
			total += pow(dice-mdoc, 2)

	sddoc = total/(len(dice_list)*7-1)
	sddoc = np.sqrt(sddoc)
	return sddoc


if __name__ == '__main__':
	pass
