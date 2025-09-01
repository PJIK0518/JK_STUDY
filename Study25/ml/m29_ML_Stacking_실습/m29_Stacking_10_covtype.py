# m10_10.copy

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

from sklearn.datasets import fetch_covtype
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
DS = fetch_covtype()

x = DS.data
y = DS.target

y = y-1

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
import warnings
import numpy as np
import random
import time

RS = 44
np.random.seed(RS)
random.seed(RS)

warnings.filterwarnings('ignore')
RS = 42

#2_1. 모델
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

XGB = XGBClassifier()
LGB = LGBMClassifier(verbosity=-1)
CAT = CatBoostClassifier(verbose=0)  # iterators Default : 1000
RFR = RandomForestClassifier()

model = [CAT, XGB, RFR, LGB]
trn_L = []
tst_L = []

for model in model :
    model.fit(x_trn, y_trn)
    
    y_trn_prd = model.predict(x_trn) # Stacking 시켜서 새로운 
    y_tst_prd = model.predict(x_tst)
    
    print(f"{model.__class__.__name__} | train pred shape: {np.shape(y_trn_prd)}, test pred shape: {np.shape(y_tst_prd)}")

    if len(y_trn_prd) == len(y_trn) and len(y_tst_prd) == len(y_tst):
        trn_L.append(np.array(y_trn_prd).reshape(-1))  # ✅ 명확히 shape 통일
        tst_L.append(np.array(y_tst_prd).reshape(-1))
    else:
        print(f"{model.__class__.__name__} 예측 길이 불일치!")

x_trn_NEW = np.array(trn_L).T
x_tst_NEW = np.array(tst_L).T

#2_2. 모델 Stacking
model_S = RandomForestClassifier()
model_S.fit(x_trn_NEW,y_trn)
y_prd_S = model_S.predict(x_tst_NEW)
score_S = accuracy_score(y_tst, y_prd_S)

print('Stacking score :', score_S)

# CatBoostClassifier | train pred shape: (522910, 1), test pred shape: (58102, 1)
# XGBClassifier | train pred shape: (522910,), test pred shape: (58102,)
# RandomForestClassifier | train pred shape: (522910,), test pred shape: (58102,)
# LGBMClassifier | train pred shape: (522910,), test pred shape: (58102,)
# Stacking score : 0.9585384324119651

# soft: 0.8763553750301194
# hard: 0.8714674193659426

# DecisionTreeRegressor  점수: 0.9435991876355375

# BaggingRegressor + DTR 점수: 0.9696568104368181 >> 직접 DTR을 Bagging 시킨 모델 : 점수 또이또이
            # bootstrap = True
            
# BaggingRegressor + DTR 점수: 0.9462324876940553
            # bootstrap = False
            # Sample 데이터 중복 허용
            
# RandomForestRegressor  점수: 0.84665 >> DTR이 Bagging 되어있는 모델  : 점수 또이또이       