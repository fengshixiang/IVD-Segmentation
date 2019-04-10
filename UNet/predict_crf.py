from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import tensorflow as tf
import logging

import image_gen
import unet
import util
from parameter import Parameter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

para = Parameter()
root_address = para.root_address
data_address = os.path.join(root_address, 'data/test')
prediction_save_address = os.path.join(root_address, 'result/pre_image_crf')
unet_model_path = os.path.join(root_address, 'result/unet_trained/model.ckpt')
provider_path = os.path.join(root_address, 'data/test/*/*/*.npy')

if not os.path.exists(prediction_save_address):
    os.mkdir(prediction_save_address)

patient_pre_list = []
patient_label_list = []
patient_save_list = []
for addr in os.listdir(data_address):     #addr:01

    image_pre_list = []
    image_label_list = []
    image_save_list = []

    patient_addr = os.path.join(data_address, addr)     #data/test/01
    patient_save_addr = os.path.join(prediction_save_address, addr)    #pre_image/01
    if not os.path.exists(patient_save_addr):
        os.mkdir(patient_save_addr)

    for addr_1 in os.listdir(patient_addr):     #addr_1:0
        image_addr = os.path.join(patient_addr, addr_1)   #data/test/01/0
        images_save_addr = os.path.join(patient_save_addr, addr_1)   #pre_image/01/0
        if not os.path.exists(images_save_addr):
            os.mkdir(images_save_addr)

        for filename in os.listdir(image_addr):
            if 'fat.npy' in filename:
                image_pre_list.append(os.path.join(image_addr, filename))     #data/test/01/0/.._fat.npy
                image_save_list.append(os.path.join(images_save_addr, filename).replace('.npy', '_pre.npy'))   #pre_image01/0/.._fat_pre.npy
            if 'label.npy' in filename:
                image_label_list.append(os.path.join(image_addr, filename)) #data/test/01/0/.._label.npy

    patient_pre_list.append(image_pre_list)
    patient_label_list.append(image_label_list)
    patient_save_list.append(image_save_list)

print('start to predict')
generator = image_gen.shapeProvider(provider_path)
net = unet.Unet(channels=generator.channels, n_class=generator.n_class, cost = para.cost,
                cost_kwargs=dict(regularizer=para.regularizer), layers=para.layers, 
                features_root=para.features_root, training=False)
def process_data(data):
    # normalization
    np.fabs(data)
    data -= np.amin(data)
    data /= np.amax(data)
    return data

def process_labels(label):
    nx = label.shape[1]
    ny = label.shape[0]
    labels = np.zeros((ny, nx, 2), dtype=np.float32)
    labels[..., 1] = label
    labels[..., 0] = ~label
    return labels

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # Initialize variables
    sess.run(init)

    # Restore model weights from previously saved model
    net.restore(sess, unet_model_path)

    for i in range(len(patient_pre_list)):
        print('predict on {}'.format(patient_pre_list[i][0]))
        for j in range(len(patient_pre_list[i])):
            if generator.channels==4:
                path = patient_pre_list[i][j]
                fat_path = path.replace("fat", "fat")
                inn_path = path.replace("fat", "inn")
                wat_path = path.replace("fat", "wat")
                opp_path = path.replace("fat", "opp")
                fat_img = np.array(np.load(fat_path), dtype=np.float32)
                inn_img = np.array(np.load(inn_path), dtype=np.float32)
                wat_img = np.array(np.load(wat_path), dtype=np.float32)
                opp_img = np.array(np.load(opp_path), dtype=np.float32)
                img_arr = np.zeros((fat_img.shape[0], fat_img.shape[1], 4), dtype=np.float32)
                
                img_arr[...,0] = fat_img
                img_arr[...,1] = inn_img
                img_arr[...,2] = wat_img
                img_arr[...,3] = opp_img
                '''
                img_arr[...,0] = opp_img
                img_arr[...,1] = opp_img
                img_arr[...,2] = opp_img
                img_arr[...,3] = opp_img
                '''
                img_arr = process_data(img_arr)
                x_test = img_arr.reshape(1, img_arr.shape[0], img_arr.shape[1], generator.channels)
                y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], 2))
                range_arr = np.empty((x_test.shape[0], 8))
                prediction = sess.run(net.predicter, feed_dict={net.x: x_test, net.y: y_dummy, net.z:range_arr, net.keep_prob: 1.})
                mask = prediction[0,...,1]
                np.save(patient_save_list[i][j], mask)
                plt.imshow(mask, cmap='Greys_r')
                plt.savefig(patient_save_list[i][j].replace('.npy', '.png'))



