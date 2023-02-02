import numpy as np


def make_Xy(inputs,
            outputs,
            len_input,
            len_output,
            window_size,
            use_lag_y=True):
    # Optional: concat y, directly use lagged y as feat.
    if use_lag_y and outputs is not None: 
        inputs = np.concatenate([inputs, outputs], axis=1)

    inputs_cond = inputs[:,:-1]
    # get shape: (t,feat,lat,lon)-(t,out,lat,lon)
    Nt, Nf, Nlat, Nlon = inputs.shape
    _, No, _, _ = outputs.shape

    # caculate the last time point to generate batch
    end_idx = inputs.shape[0] - len_input - len_output - window_size

    # generate index of batch start point in order
    batch_start_idx = range(end_idx)

    # get batch_size
    batch_size = len(batch_start_idx)

    # generate inputs
    input_batch_idx = [(range(i, i + len_input)) for i in batch_start_idx]
    inputs = np.take(inputs, input_batch_idx,
                    axis=0).reshape(batch_size, len_input, Nf, Nlat, Nlon)

    inputs = np.transpose(inputs, [0, 1, 3, 4, 2])

    if outputs is not None:
        # generate outputs
        output_batch_idx = [(range(i + len_input + window_size,
                                i + len_input + window_size + len_output))
                            for i in batch_start_idx]
        outputs = np.take(outputs, output_batch_idx, axis=0). \
            reshape(batch_size,  len_output,  No, Nlat, Nlon)
        outputs = np.transpose(outputs, [0, 1, 3, 4, 2])
    
        inputs_cond = np.take(inputs_cond, output_batch_idx, axis=0).\
            reshape(batch_size, len_output, Nf-1, Nlat, Nlon)
        inputs_cond = np.transpose(inputs_cond, [0, 1, 3, 4, 2])
    else:
        outputs=None
        inputs_cond=None
    print(inputs.shape, outputs.shape, inputs_cond.shape)
    return inputs, outputs, inputs_cond



class DataLoader():
    """Generate input for ConvLSTM"""
    def __init__(self, configs):
        self.len_input = configs.len_input
        self.len_output = configs.len_out
        self.window_size = configs.window_size
        self.use_lag_y = configs.use_lag_y

    def __call__(self, X, y):
        """run process

        NOTE: X,y must be list as X=[x_train,x_test,...], 
              y=[y_train,y_test,...], and only handle for
              ConvLSTM inputs as (t,feat,lat,lon). It 
              processes raw inputs in pair in X,y. This 
              will return input for ConvLSTM models as (t,
              len_in,lat,lon,feat)-(t,len_out,lat,lon,feat)
              in two pair list.
        """
        z = []
        for i in range(len(X)):
            _x, _y, _z = make_Xy(X[i], 
                             y[i],
                             self.len_input, 
                             self.len_output, 
                             self.window_size,
                             self.use_lag_y)
            X[i], y[i] = _x, _y
            z.append(_z)
        return X, y, z
