# m10_09.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#       >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#       >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#       >> 데이터의 소실 없이 훈련 가능

import warnings

warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

#1. 데이터
DS = load_wine()

x = DS.data
y = DS.target

from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import r2_score

# x = x.values
# y = y.values

outlier = EllipticEnvelope(contamination=.1)

outlier.fit(x)
results = outlier.predict(x)

out = []

for i in range(len(x)):
    if results[i] == 1:
        continue
    else:
        out.append(i)
        
# x = x.drop(out, axis=0)
# y = y.drop(out, axis=0)

x = np.delete(x, out, axis=0)
y = np.delete(y, out, axis=0)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777,
    stratify=y
)

#2 모델구성
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('R2 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

score1 = r2_score(y_tst, y_prd)

print('R2 :',score1)
# Outlier 처리 EE
# R2 : 0.9656997327694166

# Outlier 처리 IQR
# R2 : 0.8993897305817133
# R2 : 0.8993897306640561
#  ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.6666666666666666
# ACC : 0.6666666666666666
# F1S : 0.6031746031746031   
    
# 최적 컬럼 : 3 개 
#  ['color_intensity' 'od280/od315_of_diluted_wines' 'proline']
# 삭제 컬럼 : 10 개 
#  ['alcohol' 'malic_acid' 'ash' 'alcalinity_of_ash' 'magnesium'
#  'total_phenols' 'flavanoids' 'nonflavanoid_phenols' 'proanthocyanins'
#  'hue']
# 최고 점수 100.000%