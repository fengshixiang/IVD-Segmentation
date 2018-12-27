from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import logging

plt.rcParams['image.cmap'] = 'gist_earth'

import image_gen
import unet
import util
from parameter import Parameter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


para = Parameter()
root_address = para.root_address
generator_address = os.path.join(root_address, 'data/train/*/*/*.npy')
generator = image_gen.fourChannelProvider(generator_address)
result_path = os.path.join(root_address, 'result')
if not os.path.exists(result_path):
    os.mkdir(result_path)
unet_trained_path = os.path.join(result_path, 'unet_trained')
prediction_address = os.path.join(result_path, 'each_epoch')

#train

net = unet.Unet(channels=generator.channels, n_class=generator.n_class, cost = para.cost,
                cost_kwargs=dict(regularizer=para.regularizer), layers=para.layers, 
                features_root=para.features_root, training=True)

#trainer = unet.Trainer(net, batch_size=para.batch_size, optimizer="momentum",
#                       opt_kwargs=dict(momentum=para.momentum, learning_rate=para.learning_rate))
trainer = unet.Trainer(net, batch_size=para.batch_size, optimizer="adam",
                        opt_kwargs=dict(learning_rate=para.learning_rate, decay_rate=para.decay_rate))
path = trainer.train(generator, unet_trained_path, training_iters=para.training_iters, 
                     epochs=para.epochs, dropout=para.dropout, display_step=para.display_step, 
                     restore=para.restore, prediction_path=prediction_address)

#test one image
x_test, y_test= generator(1)
prediction = net.predict(os.path.join(unet_trained_path, 'model.ckpt'), x_test)

logging.info(
"Layers: {layers},\nFeatures: {features}\n\
Cost: {cost}, Epochs: {epochs}, Decay_epochs: {decay_epochs}\n\
Dropout: {dropout}, Learning_rate: {learning_rate}\n\
Channel: {channel}\nCreate: {create}".format(
        layers=para.layers,
        features=para.features_root,
        cost=para.cost,
        epochs=para.epochs,
        decay_epochs=para.decay_epochs,
        dropout=para.dropout,
        learning_rate=para.learning_rate,
        channel=para.channel,
        create=para.create))

fo_addr = os.path.join(root_address, 'para.txt')
fo = open(fo_addr, 'w')
fo.write("Layers: {layers}\nBatch_size: {batch_size}\nFeatures: {features}\n\
Cost: {cost}\nEpochs: {epochs}\nDecay_epochs: {decay_epochs}\nDropout: {dropout}\nLearning_rate: {learning_rate}\n\
Root_address: {root_address}\nChannel: {channel}\n\
Create: {create}".format(
            layers=para.layers,
            batch_size=para.batch_size,
            features=para.features_root,
            cost=para.cost,
            epochs=para.epochs,
            decay_epochs=para.decay_epochs,
            dropout=para.dropout,
            learning_rate=para.learning_rate,
            root_address=para.root_address,
            channel=para.channel,
            create=para.create))
fo.close()

plt.imshow(x_test[0,...,0], cmap='Greys_r')
plt.savefig(os.path.join(unet_trained_path, 'image.png'))
plt.imshow(y_test[0,...,1])
plt.savefig(os.path.join(unet_trained_path, 'label.png'))
mask = prediction[0,...,1] > para.mask
plt.imshow(mask)
plt.savefig(os.path.join(unet_trained_path, 'pre.png'))