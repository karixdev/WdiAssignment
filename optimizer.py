import numpy as np

class Optimizer:
    def __init__(self, n_iterations: int, eta=0.1) -> None:
        self.n_iterations = n_iterations
        self.eta = eta
    
    def optimize(self, theta: np.array, X: np.array, y: np.array) -> np.array:
        m = X.shape[0]

        for _ in range(self.n_iterations):
            gradient = 2 / m * X.T.dot(X.dot(theta) - y)
            theta = theta - self.eta * gradient
        
        return theta