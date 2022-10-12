import sys; sys.path.append('../V2/')
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from callback import CallBacks
from models.model_factory import model
from data_generator import DataGenerator

def train(X, y, configs):
    mdl = model(configs)
    mdl.compile(Adam(learning_rate=configs.learning_rate), loss='mse')
    history = mdl.fit(X, y,
            batch_size=configs.batch_size,
            epochs=configs.epochs,
            callbacks=[CallBacks(configs.outputs_path+'saved_model/'+configs.model_name+'/')()],
            validation_split=configs.validation_split)
    return np.concatenate([history.history["loss"],history.history["val_loss"]],axis=-1)

def train_generator(X, y, configs):
    mdl = model(configs)
    mdl.compile(Adam(learning_rate=configs.learning_rate), loss='mse')
    Ns = X.shape[0]
    n = int(Ns*0.8)
    train_generator = DataGenerator(X[:n], y[:n], configs.batch_size)
    valid_generator = DataGenerator(X[n:], y[n:], configs.batch_size)
    history = mdl.fit(train_generator,
            epochs=configs.epochs,
            callbacks=CallBacks(configs.outputs_path+'saved_model/'+configs.model_name+'/')(),
            validation_data=valid_generator)
    return np.concatenate([history.history["loss"],history.history["val_loss"]],axis=-1)

def predict(X, configs):
    md = model(configs)
    md.load_weights(configs.outputs_path+'saved_model/'+configs.model_name+'/'+str(configs.id)+'/')
    y_pred = md.predict(X)
    return y_pred

