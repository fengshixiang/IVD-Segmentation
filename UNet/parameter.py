from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os


class Parameter(object):
    def __init__(self): 
        #self.root_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/pianyi_left'
        self.root_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/010611_2'
        self.cost = 'shape_3+EPE_4' #name of the cost function. cross_entropy , dice, shape
        self.regularizer=None       #power of the L2 regularizers added to the loss function
        self.channel=4
        self.layers=5
        self.features_root=32
        self.batch_size=4
        self.training_iters=117
        self.epochs=110
        self.display_step=117*8     #number of steps till outputting stats
        self.dropout=1.0
        self.restore=False       #Flag if previous model should be restored
        self.learning_rate=0.0001
        self.decay_rate=0.1
        self.decay_epochs=80
        self.mask=0.5
        self.create=1           # 1:conventional