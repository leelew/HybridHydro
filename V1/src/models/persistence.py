from sklearn.linear_model import LinearRegression
import numpy as np


class Persistence():
    """A simple forecast based on grid-wise linear extrapolation."""
    def __init__(self):
        self.mdl = LinearRegression()

    def fit(self, X, y):
        self.T, self.C, self.H, self.W, _ = X.shape
        for i in range(self.H):
            for j in range(self.W):
                self.mdl.fit(X[:, :, i, j, 0], y[:, :, i, j, 0])

    def predict(self, X):
        y = np.full((self.T, self.C, self.H, self.W, 1), np.nan)
        for i in range(self.H):
            for j in range(self.W):
                y[:, :, i, j, 0] = self.mdl.predict(X[:, :, i, j, 0])
        return y





