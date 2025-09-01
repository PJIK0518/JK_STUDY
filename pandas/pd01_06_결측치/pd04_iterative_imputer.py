### 결측치 처리는 그때그때 컬럼을 분석을 해서 본인이 넣어야한다
### 평균 중위 앞 or 뒷 값으로 진행가능
### 가급적이면 컬럼별로!!
### interpolate : 선형보간을 이용해 적용가능
###                 >> 선형 보간의 경우에는 연속형 데이터에는 괜찮지

import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']

from sklearn.experimental import enable_iterative_imputer   # 버전따라 필요 할 수 있음
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
#################### IterativeImputer : Default ####################
imputer = IterativeImputer()            
data_II = imputer.fit_transform(data)
""" print(data_II) IterativeImputer() : BayesianRidge 회귀모델을 기반(Default)
[[ 2.          2.          2.          2.0000005 ]
 [ 4.00000099  4.          4.          4.        ]
 [ 6.          5.99999928  6.          5.9999996 ]
 [ 8.          8.          8.          8.        ]
 [10.          9.99999872 10.          9.99999874]] """

#################### IterativeImputer : XGBRegr ####################
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
RFR = RandomForestRegressor(max_depth=5,
                            random_state=333)

DTR = DecisionTreeRegressor(max_depth=5,
                            random_state=333)

XGB = XGBRegressor(max_depth = 5,
                   learning_rate = 0.1,
                   random_state = 333)

IimputerX = IterativeImputer(estimator=XGB,
                             max_iter=10,
                             random_state=333)
data_IIX = IimputerX.fit_transform(data)
""" print(data_IIX)
[[ 2.          2.          2.          4.01184034]
 [ 2.02664208  4.          4.          4.        ]
 [ 6.          4.0039463   6.          4.01184034]
 [ 8.          8.          8.          8.        ]
 [10.          7.98026466 10.          7.98815966]] """





