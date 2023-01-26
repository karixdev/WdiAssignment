import numpy as np
from model import LinearRegression
from optimizer import Optimizer
import matplotlib.pyplot as plt

np.random.seed(0)
X = 3 * np.random.rand(100, 1)

np.random.seed(1)
y = np.random.randn(100, 1) + 4 + 3 * X

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X, y)
plt.show()

model = LinearRegression()
optimizer = Optimizer(1000, 0.1)

model.fit(X, y, optimizer)

X_new = np.array([[0], [3]])
y_predicted = model.predict(X_new)

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X, y)
plt.plot(X_new, y_predicted, 'r', label="Prosta regresji")
plt.legend(loc="upper left")
plt.show()