from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import numpy as np
import random

plt.rcParams['font.family'] = 'Malgun Gothic'

RS = 518
np.random.seed(RS)
random.seed(RS)

#1. 데이터
x = 2 * np.random.rand(100, 1) -1

print(np.min(x), np.max(x)) # -0.989999972111 0.99998745111

y = 3 * x**2 + 2 * x + 1 + np.random.rand(100, 1)
# y = 3x^2 + 2x + 1 + noise

PF = PolynomialFeatures(degree=2, include_bias=False)

x_PF = PF.fit_transform(x)

print(x_PF)

#2. 모델
model_1 = LinearRegression() # y = wx + b 형태의 모델
model_2 = LinearRegression()

#3. 훈련
model_1.fit(x, y)
model_2.fit(x_PF, y)

### 그래프 그리기
plt.scatter(x, y, color='blue', label = 'Org_data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression 예제')
plt.show() 

### 다항식 회귀 그래프 그리기
x_tst = np.linspace(-1, 1, 100).reshape(-1, 1)
x_tst_PF = PF.transform(x_tst)

y_plot = model_1.predict(x_tst)
y_plot_PF = model_2.predict(x_tst_PF)

plt.plot(x_tst, y_plot, color = 'red', label ='Ogr_prd')
plt.plot(x_tst, y_plot_PF, color = 'green', label ='PF_prd')

plt.legend()
plt.grid()
plt.show()

# 실제 데이터가 다항식 형태일 때, 선형 함수로 학습하는 모델