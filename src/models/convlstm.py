import tensorflow as tf
from tensorflow.keras import Input, backend
from tensorflow.keras.layers import (Lambda, ConvLSTM2D, Input, Dense, GlobalAveragePooling2D,
                                     Multiply)
from tensorflow.keras.models import Model


def convlstm(configs):
    # default encoder-decoder ConvLSTM
    inputs = Input(shape=(configs.len_input, 
                         configs.h_w, 
                         configs.h_w, 
                         configs.n_forcing_feat+1))

    # encoder
    outputs = ConvLSTM2D(
                 8*configs.n_filters_factor, 
                 configs.kernel_size, 
                 padding=configs.padding, 
                 kernel_initializer=configs.kernel_initializer, 
                 return_state=False,
                 activation='elu',
                 recurrent_initializer='orthogonal',
                 return_sequences=True,
                 dropout=configs.dropout_rate)(inputs)

    if configs.stats_hs > 0:
        for i in range(configs.stats_hs):
            outputs = ConvLSTM2D(
                8*configs.n_filters_factor,
                configs.kernel_size,
                padding=configs.padding,
                kernel_initializer=configs.kernel_initializer,
                return_state=False,
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=True,
                dropout=configs.dropout_rate)(outputs)

    outputs, h, c = ConvLSTM2D(
                8*configs.n_filters_factor, 
                configs.kernel_size, 
                padding=configs.padding, 
                kernel_initializer=configs.kernel_initializer, 
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=False,
                dropout=configs.dropout_rate,
                return_state=True)(outputs)
    outputs = tf.expand_dims(outputs, axis=1)

    # decoder
    outs = []
    for i in range(configs.len_out):
        outputs, h, c = ConvLSTM2D(
                8*configs.n_filters_factor,
                configs.kernel_size,
                padding=configs.padding,
                kernel_initializer=configs.kernel_initializer,
                return_state=True,
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=False,
                dropout=configs.dropout_rate)(outputs, initial_state=[h, c])
        outputs = tf.expand_dims(outputs, axis=1)
        outs.append(outputs)
    outputs = tf.concat(outs, axis=1)
    out = Dense(1)(outputs)

    mdl = Model(inputs, out)
    mdl.summary()
    return mdl


def convlstm_drive(configs):
    # GFS drived encoder-decoder ConvLSTM
    inputs = Input(shape=(configs.len_input,
                          configs.h_w,
                          configs.h_w,
                          configs.n_forcing_feat+configs.n_gfs_feat+1))

    outputs = ConvLSTM2D(
        8*configs.n_filters_factor,
        configs.kernel_size,
        padding=configs.padding,
        kernel_initializer=configs.kernel_initializer,
        return_state=False,
        activation='elu',
        recurrent_initializer='orthogonal',
        return_sequences=True,
        dropout=configs.dropout_rate)(inputs)

    if configs.stats_hs > 0:
        for i in range(configs.stats_hs):
            outputs = ConvLSTM2D(
                8*configs.n_filters_factor,
                configs.kernel_size,
                padding=configs.padding,
                kernel_initializer=configs.kernel_initializer,
                return_state=False,
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=True,
                dropout=configs.dropout_rate)(outputs)

    outputs, h, c = ConvLSTM2D(
                8*configs.n_filters_factor, 
                configs.kernel_size, 
                padding=configs.padding, 
                kernel_initializer=configs.kernel_initializer, 
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=False,
                dropout=configs.dropout_rate,
                return_state=True)(outputs)
    outputs = tf.expand_dims(outputs, axis=1)

    # decoder
    outs = []
    for i in range(configs.len_out):
        outputs, h, c = ConvLSTM2D(
                8*configs.n_filters_factor,
                configs.kernel_size,
                padding=configs.padding,
                kernel_initializer=configs.kernel_initializer,
                return_state=True,
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=False,
                dropout=configs.dropout_rate)(outputs, initial_state=[h, c])
        outputs = tf.expand_dims(outputs, axis=1)
        outs.append(outputs)
    outputs = tf.concat(outs, axis=1)
    out = Dense(1)(outputs)

    mdl = Model(inputs, out)
    mdl.summary()
    return mdl


def convlstm_condition(configs):
    # GFS conditioned encoder-decoder convlstm
    inputs = Input(shape=(configs.len_input,
                          configs.h_w,
                          configs.h_w,
                          configs.n_forcing_feat+1))  # 7, 112, 112, ..
    inputs_cond = Input(shape=(configs.len_out,
                               configs.h_w,
                               configs.h_w,
                               configs.n_gfs_feat))  # 16, 112, 112, 3

    # encoder
    outputs = ConvLSTM2D(
        8*configs.n_filters_factor,
        configs.kernel_size,
        padding=configs.padding,
        kernel_initializer=configs.kernel_initializer,
        return_state=False,
        activation='elu',
        recurrent_initializer='orthogonal',
        return_sequences=True,
        dropout=configs.dropout_rate)(inputs)

    if configs.stats_hs > 0:
        for i in range(configs.stats_hs):
            outputs = ConvLSTM2D(
                8*configs.n_filters_factor,
                configs.kernel_size,
                padding=configs.padding,
                kernel_initializer=configs.kernel_initializer,
                return_state=False,
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=True,
                dropout=configs.dropout_rate)(outputs)

    outputs, h, c = ConvLSTM2D(8*configs.n_filters_factor,
                               configs.kernel_size,
                               padding=configs.padding,
                               kernel_initializer=configs.kernel_initializer,
                               activation='elu',
                               recurrent_initializer='orthogonal',
                               return_sequences=True,
                               dropout=configs.dropout_rate,
                               return_state=True)(outputs)

    # decoder
    outputs = ConvLSTM2D(8*configs.n_filters_factor,
                         configs.kernel_size,
                         padding=configs.padding,
                         kernel_initializer=configs.kernel_initializer,
                         return_state=False,
                         recurrent_initializer='orthogonal',
                         return_sequences=True,
                         dropout=configs.dropout_rate,
                         activation='elu')(inputs_cond, initial_state=[h, c])
    out = Dense(1)(outputs)

    mdl = Model([inputs, inputs_cond], out)
    mdl.summary()
    return mdl


def convlstm_se_att_condition(configs):
    # GFS att-conditioned encoder-decoder convlstm
    inputs = Input(shape=(configs.len_input,
                          configs.h_w,
                          configs.h_w,
                          configs.n_forcing_feat+1))  # 7, 112, 112, ..
    inputs_cond = Input(shape=(configs.len_out,
                               configs.h_w,
                               configs.h_w,
                               configs.n_gfs_feat))  # 16, 112, 112, 3

    # encoder
    outputs = ConvLSTM2D(
        8*configs.n_filters_factor,
        configs.kernel_size,
        padding=configs.padding,
        kernel_initializer=configs.kernel_initializer,
        return_state=False,
        activation='elu',
        recurrent_initializer='orthogonal',
        return_sequences=True,
        dropout=configs.dropout_rate)(inputs)

    if configs.stats_hs > 0:
        for i in range(configs.stats_hs):
            outputs = ConvLSTM2D(
                8*configs.n_filters_factor,
                configs.kernel_size,
                padding=configs.padding,
                kernel_initializer=configs.kernel_initializer,
                return_state=False,
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=True,
                dropout=configs.dropout_rate)(outputs)

    outputs, h, c = ConvLSTM2D(8*configs.n_filters_factor,
                               configs.kernel_size,
                               padding=configs.padding,
                               kernel_initializer=configs.kernel_initializer,
                               activation='elu',
                               recurrent_initializer='orthogonal',
                               return_sequences=True,
                               dropout=configs.dropout_rate,
                               return_state=True)(outputs)

    # decoder
    outs = []
    for i in range(configs.len_out):
        x_cond = inputs_cond[:, i:i+1]  # condition input
        x_h = tf.expand_dims(Dense(1)(h), axis=1)  # dl forecast
        x = tf.concat([x_cond, x_h], axis=-1)  # concat

        # SE-block
        x_att = GlobalAveragePooling2D()(x[:,0])  # eq.A1
        beta = Dense(x.shape[-1]//2)(x_att)  # eq.A2
        beta = Dense(x.shape[-1])(beta)
        x_att = Dense(1)(Multiply()([x, beta]))  # eq.A3

        # give weight for (f(x), x, att(x)) 
        x = tf.concat([x, x_att], axis=-1)  # f(x)+x, residual block
        x = Dense(1)(x)  # activate

        outputs, h, c = ConvLSTM2D(8*configs.n_filters_factor,
                                   configs.kernel_size,
                                   padding=configs.padding,
                                   kernel_initializer=configs.kernel_initializer,
                                   return_state=True,
                                   recurrent_initializer='orthogonal',
                                   return_sequences=True,
                                   dropout=configs.dropout_rate,
                                   activation='elu',)(x, initial_state=[h, c])
        outs.append(outputs)
    outputs = tf.concat(outs, axis=1)
    out = Dense(1)(outputs)

    mdl = Model([inputs, inputs_cond], out)
    mdl.summary()
    return mdl


def convlstm_linear_att_condition(configs):
    # GFS att-conditioned encoder-decoder convlstm
    inputs = Input(shape=(configs.len_input,
                          configs.h_w,
                          configs.h_w,
                          configs.n_forcing_feat+1))  # 7, 112, 112, ..
    inputs_cond = Input(shape=(configs.len_out,
                               configs.h_w,
                               configs.h_w,
                               configs.n_gfs_feat))  # 16, 112, 112, 3

    # encoder
    outputs = ConvLSTM2D(
        8*configs.n_filters_factor,
        configs.kernel_size,
        padding=configs.padding,
        kernel_initializer=configs.kernel_initializer,
        return_state=False,
        activation='elu',
        recurrent_initializer='orthogonal',
        return_sequences=True,
        dropout=configs.dropout_rate)(inputs)

    if configs.stats_hs > 0:
        for i in range(configs.stats_hs):
            outputs = ConvLSTM2D(
                8*configs.n_filters_factor,
                configs.kernel_size,
                padding=configs.padding,
                kernel_initializer=configs.kernel_initializer,
                return_state=False,
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=True,
                dropout=configs.dropout_rate)(outputs)

    outputs, h, c = ConvLSTM2D(8*configs.n_filters_factor,
                               configs.kernel_size,
                               padding=configs.padding,
                               kernel_initializer=configs.kernel_initializer,
                               activation='elu',
                               recurrent_initializer='orthogonal',
                               return_sequences=True,
                               dropout=configs.dropout_rate,
                               return_state=True)(outputs)

    # decoder
    outs = []
    for i in range(configs.len_out):
        # NOTE: This is a simplify edition of attention condition ConvLSTM.
        #       We remove SE part, and only give weight for (f(x), x).
        #       Preliminary test shows that this model slightly worse than
        #       SE attention condition ConvLSTM. 
        x_cond = inputs_cond[:, i:i+1]  # condition input
        x_h = tf.expand_dims(Dense(1)(h), axis=1)  # dl forecast
        x = tf.concat([x_cond, x_h], axis=-1)  # concat
        x = Dense(1)(x)  # linear attention

        outputs, h, c = ConvLSTM2D(8*configs.n_filters_factor,
                                   configs.kernel_size,
                                   padding=configs.padding,
                                   kernel_initializer=configs.kernel_initializer,
                                   return_state=True,
                                   recurrent_initializer='orthogonal',
                                   return_sequences=True,
                                   dropout=configs.dropout_rate,
                                   activation='elu',)(x, initial_state=[h, c])
        outs.append(outputs)
    outputs = tf.concat(outs, axis=1)
    out = Dense(1)(outputs)

    mdl = Model([inputs, inputs_cond], out)
    mdl.summary()
    return mdl
