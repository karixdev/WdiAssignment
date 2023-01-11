import numpy as np

np.random.seed(0)
X = np.random.rand(2, 1)

np.random.seed(1)
y = np.random.randn(2, 1) + 4 + 3 * X

print(X)
print(y)