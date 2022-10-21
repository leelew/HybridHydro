import sys; sys.path.append('../../src/')
import numpy as np
from tensorflow.keras.optimizers import Adam

from callback import CallBacks
from loss import MaskMSELoss
from models.model_factory import model


def train(X, y, configs, mask=None):
    mdl = model(configs)
    # load V2 global model before.
    mdl.load_weights(configs.outputs_path+'saved_model/'+configs.model_name+'_V2/')
    mdl.compile(Adam(learning_rate=configs.learning_rate), loss=MaskMSELoss(mask))
    history = mdl.fit(X, y,
            batch_size=configs.batch_size,
            epochs=configs.epochs,
            callbacks=[CallBacks(configs.outputs_path+'saved_model/'+configs.model_name+'/'+str(configs.id)+'/')()],
            validation_split=configs.validation_split)
    return np.stack([history.history["loss"],history.history["val_loss"]],axis=-1)


def predict(X, configs):
    mdl = model(configs)
    mdl.load_weights(configs.outputs_path+'saved_model/'+configs.model_name+'/'+str(configs.id)+'/')
    y_pred = mdl.predict(X)
    return y_pred


if __name__ == '__main__':
    train()
