import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def Mish(x):
#     return x * np.tanh(np.log(1+np.exp(x)))

Mish = lambda x : x * np.tanh(np.log(1+np.exp(x)))

y = Mish(x)

plt.plot(x, y)
plt.grid()
plt.show()

# Relu 마냥 0 이하를 처리하는데 애들이 성능이 좋아질 수도 있음
# But. 데이터가 커지면 연산량이 증가하는 문제...