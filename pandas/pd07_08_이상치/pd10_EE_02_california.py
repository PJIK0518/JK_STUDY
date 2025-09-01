import numpy as np
import random
import warnings

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')

RS = 35
np.random.seed(RS)
random.seed(RS)

#1. 데이터
DS = fetch_california_housing()
x = DS.data
y = DS.target

from sklearn.covariance import EllipticEnvelope
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
    train_size=0.8,
    random_state=RS,  
)

#2 모델구성
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('R2 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

score1 = r2_score(y_tst, y_prd)
print('# Outlier 처리')
print('# R2 :',score1)
# Outlier 처리_EE
# R2 : 0.6627049599408086

# Outlier 처리_IQR
# R2 : 0.6656905139551408

# PF
# 02 R2 : 0.665690514019535

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