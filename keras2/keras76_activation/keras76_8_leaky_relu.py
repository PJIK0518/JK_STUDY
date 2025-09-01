import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def leaky_relu(x, alpha):
    # return np.maximum(alpha*x, x)
    # return np.where(x > 0, x, alpha*x)
                # np.where(a, b, c) : a 조건일때, 참 이면 b를 실행, 거짓이면, c를 실행 

# leaky_relu = lambda x, alpha : np.maximum(alpha*x, x)
leaky_relu = lambda x, alpha : np.where(x > 0, x, alpha*x)

y = leaky_relu(x, 0.01)

plt.plot(x, y)
plt.grid()
plt.show()

# alpha 값으로 최소값을 조절할 수 있는녀석
