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

PF = PolynomialFeatures(degree=5,
                        include_bias=False
                        )

x_PF = PF.fit_transform(x)

print(x_PF)