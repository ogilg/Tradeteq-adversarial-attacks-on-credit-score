import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

class LogScaler(TransformerMixin):

    def __init__(self, scale=True):
        self.scale = scale

    def fit(self, X, y=None):
        if self.scale:
            X_log = np.array(np.log10(X+1))
            X_log = X_log.reshape(-1,1)
            self.scaler = StandardScaler().fit(X_log)
        return self

    def transform(self, X):
        X_log = np.array(np.log10(X+1))
        if self.scale:
            X_log = X_log.reshape(-1, 1)
            return self.scaler.transform(X_log)
        else:
            return X_log

