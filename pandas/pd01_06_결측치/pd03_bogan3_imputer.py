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
""" print(data)
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN """

from sklearn.experimental import enable_iterative_imputer   # 버전따라 필요 할 수
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

Simputer = SimpleImputer(strategy='constant', fill_value=777)
data_SI = Simputer.fit_transform(data)
""" print(data_SI) : Default / SimpleImputer() = strategy='mean'
[[ 2.          2.          2.          6.        ]
 [ 6.5         4.          4.          4.        ]
 [ 6.          4.66666667  6.          6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]] """
""" print(data_SI) : SimpleImputer(strategy='mean')
[[ 2.          2.          2.          6.        ]
 [ 6.5         4.          4.          4.        ]
 [ 6.          4.66666667  6.          6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]]
"""
""" print(data_SI) : SimpleImputer(strategy='median')
[[ 2.  2.  2.  6.]   median : 데이터를 정렬해서 중간 위치의 값
 [ 7.  4.  4.  4.]
 [ 6.  4.  6.  6.]
 [ 8.  8.  8.  8.]
 [10.  4. 10.  6.]] """

####################### most_frequent #######################
data_f = pd.DataFrame([[2, np.nan, 6, 8, 10, 8],
                     [2, 4, np.nan, 8, np.nan, 4],
                     [2, 4, 6, 8, 10, 12],
                     [np.nan, 4, np.nan, 8, np.nan, 8]])

data_f = data_f.transpose()
data_f.columns = ['x1', 'x2', 'x3', 'x4']
data_f_SI = Simputer.fit_transform(data_f)
""" print(data_f_SI) : SimpleImputer(strategy='most_frequent')
[[ 2.  2.  2.  8.] : 최빈값으로 결측치 처리
 [ 8.  4.  4.  4.] : 자연적인 결측치가 발생한 
 [ 6.  4.  6.  8.]
 [ 8.  8.  8.  8.]
 [10.  4. 10.  8.]
 [ 8.  4. 12.  8.]] """
#############################################################

""" print(data_SI) : SimpleImputer(strategy='constant', fill_value=777)
[[  2.   2.   2. 777.] : 임의로 지정한 특정 상수로 결측치 처리
 [777.   4.   4.   4.]
 [  6. 777.   6. 777.]
 [  8.   8.   8.   8.]
 [ 10. 777.  10. 777.]] """

######################### KNNImputer ########################
Kimputer = KNNImputer()
data_KI = Kimputer.fit_transform(data)
""" print(data_KI) : Kimputer.fit_transform(data) 평균값
[[ 2.          2.          2.          6.        ]
 [ 6.5         4.          4.          4.        ]
 [ 6.          4.66666667  6.          6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]] """
 
###################### IterativeImputer #####################
Iimputer = IterativeImputer()
date_II = Iimputer.fit_transform(data)
""" print(date_II) : IterativeImputer() : 다른 컬럼을 기반으로 해당 열의 결측치 계산
                                        : 모델 기반 - Default : BayesianRidge 회귀모델을 기반
[[ 2.          2.          2.          2.0000005 ]
 [ 4.00000099  4.          4.          4.        ]
 [ 6.          5.99999928  6.          5.9999996 ]
 [ 8.          8.          8.          8.        ]
 [10.          9.99999872 10.          9.99999874]] """