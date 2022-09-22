import numpy as np
from data.ops.make_X_y import make_Xy


class DataLoader():
    #[(sample, in_len, lat, lon, f), (sample, out_len, lat, lon, 1)]

    def __init__(self,
                 len_input,
                 len_output,
                 window_size,
                 use_lag_y=True,
                 mode='train'):
        self.len_input = len_input
        self.len_output = len_output
        self.window_size = window_size
        self.use_lag_y = use_lag_y
    
    def fit_teacher_forcing(self, y):
        """generate inputs for teacher forcing training.

        Args:
            y ([type]): shape as (S, 1, 112, 112)
        
        Outputs:
            X: shape as (S-14, 14, 112, 112, 1)
            y: shape as (S-14, 14, 112, 112, 1)
        """
        X, y = self.make_tf_Xy(y, len_input=7, len_output=7, window_size=6)
        return X, y


    def make_tf_Xy(
            self,
            inputs,
            len_input=7,
            len_output=7,
            window_size=0):

        Nt, _, Nlat, Nlon = inputs.shape
        """Generate inputs and outputs for LSTM."""
        # caculate the last time point to generate batch
        end_idx = inputs.shape[0] - len_input - len_output - window_size - 1
        print(end_idx)
        # generate index of batch start point in order
        batch_start_idx = range(end_idx)
        # get batch_size
        batch_size = len(batch_start_idx)
        print(batch_size)

        # generate inputs
        input_batch_idx = [(range(i, i + len_input)) for i in batch_start_idx]
        X = np.take(inputs, input_batch_idx, axis=0).reshape(batch_size, len_input, inputs.shape[-1], Nlat, Nlon)

        X = np.transpose(X, [0, 1, 3, 4, 2])
        print(X.shape)

        # generate outputs
        output_batch_idx = [(range(i + len_input + window_size,
                                i + len_input + window_size + len_output))
                            for i in batch_start_idx]
        y = np.take(inputs, output_batch_idx, axis=0). \
            reshape(batch_size,  len_output,  1, Nlat, Nlon)
        y = np.transpose(y, [0, 1, 3, 4, 2])
        print(y.shape)
        return X, y   


    def __call__(self, X, y):
        """[summary]

        Args:
            X ([type]): (samples, timestep, height, width, features)
            y ([type]): (samples, timestep, height, width, 1)
            z ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        # generate inputs
        for i in range(len(X)):
            _x, _y = make_Xy(X[i], y[i],
                             self.len_input, 
                             self.len_output, 
                             self.window_size,
                             self.use_lag_y)
            X[i], y[i] = _x, _y
        return X, y
