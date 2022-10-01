import sys; sys.path.append('../../src/')
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from factory.callback import CallBacks#, TuneReporterCallback
from factory.loss import MaskMSELoss
from models.model_factory import model

def train(X, y, configs, mask=None):

    #FIXME: Maybe not need?
    #       make tensorflow dataset.
    dataset = tf.data.Dataset.\
        from_tensor_slices((X, y)).\
        shuffle(configs.shuffle_times).\
        batch(configs.batch_size, drop_remainder=True)

    mdl = model(configs)
    #FIXME: Give parmaeters for different loss
    mdl.compile(Adam(learning_rate=configs.learning_rate), loss=MaskMSELoss(mask))
    mdl.fit(X, y,
            batch_size=configs.batch_size,
            epochs=configs.epochs,
            callbacks=[CallBacks(configs.ROOT+configs.saved_model_path+str(configs.id)+'/')()],#,TuneReporterCallback()],
            validation_split=configs.validation_split)

def predict(X, configs):
    mdl = model(configs)
    mdl.load_weights(configs.ROOT+configs.saved_model_path+str(configs.id)+'/')
    y_pred = mdl.predict(X)
    return y_pred


if __name__ == '__main__':
    train()
