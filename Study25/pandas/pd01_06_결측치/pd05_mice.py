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

import numpy as np

# NumPy 1.24+ 호환성 패치
if not hasattr(np, "float"):
     
    np.float = float   # 또는 np.float64
    
# from sklearn.experimental import enable_iterative_imputer   # 버전따라 필요 할 수 있음
# from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
# from xgboost import XGBRegressor

from impyute.imputation.cs import mice
data_M = mice(data.values,
              n=10,
              seed=518)
""" print(data_M) # itrative 계열 : n번의 선형회귀 후 약간의 noise 부여
[[ 2.          2.          2.          1.98390121]
[ 4.03228561  4.          4.          4.        ]
[ 6.          6.01351212  6.          6.02607766]
[ 8.          8.          8.          8.        ]
[10.         10.05404847 10.         10.13114195]] """

### 결측치는 다른 데이터를 보고 판단 및 분석이 가능하면 적용
### 특정 수치를 넣거나,