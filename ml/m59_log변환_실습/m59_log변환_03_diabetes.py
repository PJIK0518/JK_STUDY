from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

seed = 123
random.seed(seed)
np.random.seed(seed)
    
#1 데이터
DS =load_diabetes()
x = DS.data
y = DS.target

print(x.shape, y.shape)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    random_state=seed)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# x_trn = np.log1p(x_trn)
# x_tst = np.log1p(x_tst)

y_trn = np.log1p(y_trn)
y_tst = np.log1p(y_tst)

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
#2 모델구성
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score
model = RandomForestRegressor()     # k_neighbors랑 같은 Parameter

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('R2 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)
y_tst = np.expm1(y_tst)
y_prd = np.expm1(y_prd)
score1 = r2_score(y_tst, y_prd)

print('R2 :',score1)
## RFR
# R2 : 0.4397566348634161
# R2 : 0.4397566348634161

## x, y 변환
# R2 : 0.460535482522633
# R2 : 0.4167998674346418

## x만 변환
# R2 : 0.4388756297932651
# R2 : 0.4388756297932651

## y만 변환
# R2 : 0.46401376904942404
# R2 : 0.42138428809454365