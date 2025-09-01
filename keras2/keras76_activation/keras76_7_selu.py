import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def selu(x, alpha, lmbda):
#     return lmbda * ((x>0)*x + (x<=0)*(alpha*(np.exp(x)-1)))

selu = lambda x, alpha, lmbda : lmbda * ((x>0)*x + (x<=0)*(alpha*(np.exp(x)-1)))

y = selu(x, 1.67, 1.05)
y = selu(x, 2, 5)

plt.plot(x, y)
plt.grid()
plt.show()

# alpha 값으로 최소값을 조절할 수 있는녀석
