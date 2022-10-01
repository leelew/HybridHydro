import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, backend
from tensorflow.keras.layers import ConvLSTM2D, Input, Lambda
from tensorflow.keras.models import Model


def convlstm_v1(configs):
    inputs = Input(shape=configs.convlstm_inputs_shape)

    outputs = tf.keras.layers.ConvLSTM2D(
                8*configs.n_filters_factor, 
                configs.kernel_size, 
                padding=configs.padding, 
                kernel_initializer=configs.kernel_initializer, 
                return_state=False,
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=True,
                dropout=configs.dropout_rate)(inputs)

    if configs.stats_hs == 0:
        print('a')
    else:
        for i in range(configs.stats_hs):

            outputs = tf.keras.layers.ConvLSTM2D(
                    8*configs.n_filters_factor, 
                    configs.kernel_size, 
                    padding=configs.padding, 
                    kernel_initializer=configs.kernel_initializer, 
                    return_state=False,
                    activation='elu',
                    recurrent_initializer='orthogonal',
                    return_sequences=True,
                    dropout=configs.dropout_rate)(outputs)

    outputs = tf.keras.layers.ConvLSTM2D(
                8*configs.n_filters_factor, 
                configs.kernel_size, 
                padding=configs.padding, 
                kernel_initializer=configs.kernel_initializer, 
                return_state=False,
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=False,
                dropout=configs.dropout_rate)(outputs)

    out = Lambda(lambda x: backend.concatenate([x] * configs.len_out, axis=1))(outputs[:, tf.newaxis])
    out = ConvLSTM2D(8*configs.n_filters_factor, 
                     configs.kernel_size, 
                     padding='same', 
                     return_sequences=True,
                     activation='elu',
                     recurrent_initializer='orthogonal',
                     dropout=configs.dropout_rate)(out)
        
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model(inputs, out)
    mdl.summary()    
    return mdl


def convlstm_teacher_forcing(configs):
    inputs = Input(shape=configs.convlstm_inputs_shape) # 7, 112, 112, 8
    inputs_tf = Input(shape=configs.gfs_inputs_shape) # 16, 112, 112, 3


    outputs = tf.keras.layers.ConvLSTM2D(
                8*configs.n_filters_factor, 
                configs.kernel_size, 
                padding=configs.padding, 
                kernel_initializer=configs.kernel_initializer, 
                return_state=False,
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=True,
                dropout=configs.dropout_rate)(inputs)
    # 7， 112， 112， 8

    outputs, h, c = ConvLSTM2D(8*configs.n_filters_factor, 
                configs.kernel_size, 
                padding=configs.padding, 
                kernel_initializer=configs.kernel_initializer, 
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=True,
                dropout=configs.dropout_rate,
                return_state=True)(outputs)
    # 1， 112， 112， 8

    #states = tf.transpose(states, [0, 4, 2, 3, 1])
    #states = Dense(1)(states)
    #states = tf.transpose(states, [0, 4, 2, 3, 1])
    print('1')
    print(outputs.shape)



    out = ConvLSTM2D(8*configs.n_filters_factor, 
                configs.kernel_size, 
                padding=configs.padding, 
                kernel_initializer=configs.kernel_initializer, 
                return_state=False,
                activation='elu',
                recurrent_initializer='orthogonal',
                return_sequences=True,
                dropout=configs.dropout_rate)(inputs_tf, initial_state=[h, c])
    
    out = tf.keras.layers.Dense(1)(out)

    mdl = Model([inputs, inputs_tf], out)

    mdl.summary()    
    return mdl

