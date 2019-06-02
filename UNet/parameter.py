from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os


class Parameter(object):
    def __init__(self): 
        self.root_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/8modality_2'
        self.cost = 'CE' #name of the cost function. cross_entropy , dice
        self.regularizer=None       #power of the L2 regularizers added to the loss function
        self.channel=8
        self.layers=5
        self.features_root=32
        self.batch_size=4
        self.training_iters=468 // self.batch_size
        self.epochs=100
        self.display_step=468//self.batch_size*4     #number of steps till outputting stats
        self.dropout=1.0
        self.restore=False       #Flag if previous model should be restored
        self.learning_rate=0.0001
        self.decay_rate=0.5
        self.decay_epochs=80
        self.mask=0.5
        self.create=1           # 1:conventional
        self.RMVD=False
        self.RMVD_value=0.5