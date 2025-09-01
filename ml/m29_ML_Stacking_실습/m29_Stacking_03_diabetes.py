# m10_03.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#       >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#       >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#       >> 데이터의 소실 없이 훈련 가능

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
DS = load_diabetes()

x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.85,
    shuffle=True,
    random_state=777
)

MS = MinMaxScaler()

MS.fit(x_trn)

x_trn = MS.transform(x_trn)
# x_tst = MS.transform(x_tst)

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

#2_1. 모델
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

XGB = XGBRegressor()
LGB = LGBMRegressor(verbosity=-1)
CAT = CatBoostRegressor(verbose=0)
RFR = RandomForestRegressor()

model = [CAT, XGB, RFR, LGB]
trn_L = []
tst_L = []

for model in model :
    model.fit(x_trn, y_trn)
    
    y_trn_prd = model.predict(x_trn) 
    y_tst_prd = model.predict(x_tst)
    
    trn_L.append(y_trn_prd)
    tst_L.append((y_tst_prd))
    
    score = r2_score(y_tst, y_tst_prd)
    
    class_name = model.__class__.__name__
    
    print(f'{class_name} R2 : {score:.4f}')

x_trn_NEW = np.array(trn_L).T
x_tst_NEW = np.array(tst_L).T

#2_2. 모델 Stacking
model_S = RandomForestRegressor()
model_S.fit(x_trn_NEW,y_trn)
y_prd_S = model_S.predict(x_tst_NEW)
score_S = r2_score(y_tst, y_prd_S)

print('Stacking score :', score_S)

# CatBoostRegressor R2 : -0.2286
# XGBRegressor R2 : -0.0208
# RandomForestRegressor R2 : -0.2505
# LGBMRegressor R2 : -0.3978
# Stacking score : -0.019540457000954747