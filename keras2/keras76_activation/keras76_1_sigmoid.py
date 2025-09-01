import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

sigmoid = lambda x : 1 / (1 + np.exp(-x))
# lambda : 함수 설정하는 다른 방식

x = np.arange(-5, 5, 0.1)

print(x)
print(len(x))

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 이렇게 sigmoid를 먹이면 활성 함수를 넣는거보다 연산량에 부하가 덜 함..!