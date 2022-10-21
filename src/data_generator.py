import json
import math

import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    # NOTE: Need test
    def __init__(self, x, y, batch_size, shuffle=True):
        super().__init__()
        self.x, self.y = x, y  # (ngrid, nt, nfeat)-(ngrid, nt, nout)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.nsamples = x.shape[0]
        self.indexes = np.arange(self.nsamples)
    def __len__(self):
        return math.ceil(self.nsamples / self.batch_size)

    def __getitem__(self, idx):
        # index for grids
        grid_idx = self.indexes[idx * self.batch_size:(idx+1)*self.batch_size]
        # crop
        x_ = self.x[grid_idx]
        y_ = self.y[grid_idx]
        return x_, y_

    def on_epoch_end(self):
        self.indexes = np.arange(self.nsamples)
        if self.shuffle:
            np.random.shuffle(self.indexes)
