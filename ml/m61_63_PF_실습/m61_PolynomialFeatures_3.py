## SMOTE SMOTENC ROS : 행 증폭!
## 그렇다면 열 증폭은 어떻게 할까??
## 성질이 비슷한 녀석들을 더하거나 곱하거나
# > 알고리즘 형태로도 존재 한다!!

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4, 3)
print(x)

PF = PolynomialFeatures(degree=2,
                        include_bias=False,
                        interaction_only=True # 제곱을 하지 않고 각 컬럼끼리의 곱셈
                                              # 통상적으로는 성능이 덜함 >> PCA , feature importance, correlation 으로..!
                        )

x_PF = PF.fit_transform(x)

print(x_PF)