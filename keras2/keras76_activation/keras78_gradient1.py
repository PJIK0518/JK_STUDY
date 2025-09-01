import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6

x = np.linspace(-1, 6, 100) # -1~6 까지 백득분해서 값을 가져온다

print(x)
print(len(x))

y = f(x)

plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk', color = 'red')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()