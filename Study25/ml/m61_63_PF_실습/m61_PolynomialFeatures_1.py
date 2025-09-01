## SMOTE SMOTENC ROS : 행 증폭!
## 그렇다면 열 증폭은 어떻게 할까??
## 성질이 비슷한 녀석들을 더하거나 곱하거나
# > 알고리즘 형태로도 존재 한다!!

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4, 2)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]

PF = PolynomialFeatures(degree=2,
                        include_bias=False
                        )

x_PF = PF.fit_transform(x)

print(x_PF)
# include_bias = True (Default)
# degree = 2
# bias / x1 / x2 / x1^2 / x1*x2 / x2^2
#  [[1.   0.   1.     0.     0.      1.]
#   [1.   2.   3.     4.     6.      9.]
#   [1.   4.   5.    16.    20.     25.]
#   [1.   6.   7.    36.    42.     49.]]

# include_bias = False
# [[0.  1.   0.   0.   1.]
#  [2.  3.   4.   6.   9.]
#  [4.  5.  16.  20.  25.]
#  [6.  7.  36.  42.  49.]]

# degree = 3
# b / x1 /  x2
#   x1^2 / x1^2 * x2
#   x2^2 / x1 * x2^2
#   x1^3 / x2^3

#### 통상적으로...
# 선형모델 (lr등)에 쓸 경우에는 include_bias = True를 써서 1 만 있는 컬럼을 만드는게 좋음
# 왜냐하면, y = wx + b의 bias = 1의 역할을 하기 때문
# 비선형 모델 (rf , xgb)에 쓸 경우에는 include_bias = False 가 좋음

# 선형을 비선형으로 만드는 것에 효과적

# Degree 