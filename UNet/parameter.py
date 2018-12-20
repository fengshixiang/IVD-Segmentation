from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os


class Parameter(object):
    def __init__(self): 
        self.root_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/01-13_2'
        #self.root_address = '/DATA5_DB8/data/sxfeng/data/IVDNet/experiment/010616'
        self.cost = 'cross_entropy' #name of the cost function. cross_entropy , dice
        self.regularizer=None       #power of the L2 regularizers added to the loss function
        self.channel=4
        self.layers=5
        self.features_root=16
        self.batch_size=4
        self.training_iters=117
        self.epochs=100
        self.display_step=1170     #number of steps till outputting stats
        self.dropout=1.0
        self.restore=False       #Flag if previous model should be restored
        self.learning_rate=0.0001
        self.decay_rate=0.5
        self.decay_epochs=50
        self.mask=0.5
        self.create=1           # 1:conventional