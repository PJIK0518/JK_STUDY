import numpy as np
import random
import warnings

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
warnings.filterwarnings('ignore')

RS = 35
np.random.seed(RS)
random.seed(RS)

#1. 데이터
DS = fetch_california_housing()
x = DS.data
y = DS.target

import matplotlib.pyplot as plt
x_DF = pd.DataFrame(x, columns=DS.feature_names)
y_DF = pd.DataFrame(y, columns=['target'])

# x_DF.boxplot()
plt.boxplot(x_DF)
plt.show()

## 각컬럼의 데이터 분포 확인 > 이상치 존재여부 확인가능
## log화 시킬 때 이상치 보정에 훌륭

exit()

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.8,
    random_state=RS,  
)


x_trn = np.log1p(x_trn)
x_tst = np.log1p(x_tst)

y_trn = np.log1p(y_trn)
y_tst = np.log1p(y_tst)

# print(x.shape, y.shape) (20640, 8) (20640,)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb

#2 모델구성
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score
model = RandomForestRegressor()

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('R2 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

# y_tst = np.expm1(y_tst)
# y_prd = np.expm1(y_prd)

score1 = r2_score(y_tst, y_prd)

print('R2 :',score1)
# RFR
# R2 : 0.8114202090374049
# R2 : 0.8114202090374049

## x, y 변환
# R2 : 0.7395296108401057
# R2 : 0.7314014418807302

## x만 변환
# R2 : 0.7361044731472235
# R2 : 0.7361044731472235

## y만 변환
# R2 : 0.8280160231711524
# R2 : 0.8057273043107682
