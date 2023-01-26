import numpy as np
from optimizer import Optimizer

class LinearRegression:
    def fit(self, X: np.array, y: np.array, optimizer: Optimizer) -> None:
        X_new = np.c_[np.ones(X.shape), X]
        self.theta = np.zeros([X_new.shape[1], 1])

        self.theta = optimizer.optimize(self.theta, X_new, y)
    
    def predict(self, X: np.array) -> np.array:
        X_new = np.c_[np.ones([X.shape[0], 1]), X]

        return X_new.dot(self.theta)