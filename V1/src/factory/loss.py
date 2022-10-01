import tensorflow as tf
from tensorflow.keras import losses
import tensorflow.python.keras.backend as K
from tensorflow.keras.losses import MeanAbsoluteError



class ImageGradientDifferenceLoss(losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        # for 5D inputs
        gdl = 0
        for i in range(y_true.shape[1]):
            dy_true, dx_true = tf.image.image_gradients(y_true[:,i])
            dy_pred, dx_pred = tf.image.image_gradients(y_pred[:,i])
            gdl+=K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true))
        return gdl

class LPLoss(losses.Loss):
    def __init__(self, l_num=2):
        self.l_num = l_num #NOTE: tensorflow.loss must set at __init__.
        super().__init__()

    def call(self, y_true, y_pred, l_num=2):
        mse = tf.math.reduce_mean((y_true - y_pred)**self.l_num, axis=0)
        return tf.math.reduce_mean(mse)


class MaskSeq2seqLoss(losses.Loss):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def call(self, y_true, y_pred):
        pass

class MaskMSELoss(tf.keras.losses.Loss):
    def __init__(self, mask, add_ssim=False, add_gdl=False, add_mae=False):
        super().__init__()
        self.mask = mask
        self.add_ssim = add_ssim
        self.add_gdl = add_gdl
        self.add_mae = add_mae

    def call(self, y_true, y_pred):

        weight = 20*tf.ones_like(y_true)
        m = tf.where(y_true>0.3, weight , 10*weight)
        m = tf.where(y_true<0.1, m , 20*weight)


        if self.add_gdl:
            gdl = 0
            for i in range(y_true.shape[1]):
                dy_true, dx_true = tf.image.image_gradients(y_true[:,i])
                dy_pred, dx_pred = tf.image.image_gradients(y_pred[:,i])
                gdl+=K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true))

        if self.add_ssim:
            y_true_ = tf.math.multiply(y_true, self.mask)
            y_pred_ = tf.math.multiply(y_pred, self.mask)

            ssim = 1 - tf.reduce_mean(
                tf.image.ssim(y_true_, y_pred_, max_val=1.0, filter_size=3))

        if self.add_mae:
            mae = tf.math.reduce_mean(m[:, 4:]*tf.abs(y_true[:, 4:]-y_pred[:, 4:]), axis=0)
            mask_mae = tf.math.multiply(mae, self.mask)
            mask_mae = tf.math.reduce_mean(mask_mae)
            

        mse = tf.math.reduce_mean(m*tf.square(y_true - y_pred), axis=0)
        mask_mse = tf.math.multiply(mse, self.mask)

        x = tf.math.reduce_mean(mask_mse)

        if self.add_ssim: x = x+0.05*ssim
        if self.add_gdl: x = x+0.05*gdl
        if self.add_mae: x = (x+mask_mae)/2

        return x


class MaskSSIMLoss(tf.keras.losses.Loss):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def call(self, y_true, y_pred):
        y_true_ = tf.math.multiply(y_true, self.mask)
        y_pred_ = tf.math.multiply(y_pred, self.mask)

        return 1 - tf.reduce_mean(
            tf.image.ssim(y_true_, y_pred_, max_val=1.0, filter_size=3))
