import numpy as np


def _check_x(inputs):

    # check inputs by following constrains
    # 1) if any feature of inputs is all NaN, then discard this inputs.
    # 2) if dimensions of inputs is not equal to 4, then raise error.

    if inputs.ndim == 4:
        Nt, Nlat, Nlon, Nf = inputs.shape

        #for i in range(Nf):
        pass
    else:
        raise TypeError('The dimension is not equal to 4')


class CheckXy():
    """Check X,y before training.


    """
    def __init__(self, X, y):
        pass

    def __call__(self) -> Any:
        X[X < 0] = np.nan
        y[y < 0] = np.nan

        X[X > 1] = np.nan
        y[y > 1] = np.nan
