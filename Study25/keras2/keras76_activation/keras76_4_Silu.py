# 구글에서 2017에 swish 로발표

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def Silu(x):
#     return x * (1 / (1 + np.exp(-x)))

Silu = lambda x : x * (1 / (1 + np.exp(-x)))

y = Silu(x)

plt.plot(x, y)
plt.grid()
plt.show()