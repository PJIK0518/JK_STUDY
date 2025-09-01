import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

tanh = lambda x : (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

y = tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()