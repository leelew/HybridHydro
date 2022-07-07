# ==============================================================================
# Make input for single grid or multi-grids (images)
#
# author: Lu Li, 2021/09/27
# ==============================================================================


import numpy as np


def make_Xy(
        inputs,  #(t, f, lat, lon)
        outputs,
        len_input,
        len_output,
        window_size,
        use_lag_y=True):
    if use_lag_y and outputs is not None:
        inputs = np.concatenate([inputs, outputs], axis=1)
    Nt, Nf, Nlat, Nlon = inputs.shape
    _, No, _, _ = outputs.shape
    """Generate inputs and outputs for LSTM."""
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

    # 判断两种reshape方式的输入的区别？看看是不是有明显区别。没有的话看只预报一天的结果。
    if outputs is not None:

        # generate outputs
        output_batch_idx = [(range(i + len_input + window_size,
                                i + len_input + window_size + len_output))
                            for i in batch_start_idx]
        outputs = np.take(outputs, output_batch_idx, axis=0). \
            reshape(batch_size,  len_output,  No, Nlat, Nlon)
        outputs = np.transpose(outputs, [0, 1, 3, 4, 2])

    else:
        outputs=None
    return inputs, outputs

