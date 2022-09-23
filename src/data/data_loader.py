from data.ops.make_Xy import make_Xy


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
        for i in range(len(X)):
            _x, _y = make_Xy(X[i], 
                             y[i],
                             self.len_input, 
                             self.len_output, 
                             self.window_size,
                             self.use_lag_y)
            X[i], y[i] = _x, _y
        return X, y
