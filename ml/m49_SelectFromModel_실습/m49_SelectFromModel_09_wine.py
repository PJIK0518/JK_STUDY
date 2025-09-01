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
x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777,
    stratify=y
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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb

#2 모델구성
model = XGBClassifier(
    n_estimators = 200,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,                 # CHAT GPT에 각각 계산 방식 물어보기 #
 #  eval_metric = 'mlogloss',       # 다중 분류 : mlogloss, merror
                                    # 이진 분류 : logloss, error
                                    # 2.1.1 버전 이후로 fit 에서 모델로 위치 변경
    early_stopping_rounds=10,
    random_state=RS
    )

model.fit(x_trn, y_trn,
          eval_set = [(x_tst,y_tst)],
          verbose = 0)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('R2 :', model.score(x_tst, y_tst))
print(model.feature_importances_)


thresholds = np.sort(model.feature_importances_)
# 훈려에서 피처 중요도에 따라 오름차순으로 정렬

# print(thresholds)

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)
    # thresholds가 i 이상인 것을 모두 훈련 싵켜러       # prefit: 훈련을 다시 시키고 진행할 건지 결정
    # prefit = False : 모델이 아직 학습되지 않았을 때, Fit 호출해서 훈련 (Defualt)
    # prefit = True  : 이미 학습된 모델을 전달할 때, 
     
    select_x_trn = selection.transform(x_trn)
    select_x_tst = selection.transform(x_tst)
    # 순차적으로 중요도가 낮은 애들부터 하나씩 제거하면서 데이터 형성
    # print(select_x_trn.shape)
    
    select_model = XGBClassifier(
    n_estimators = 200,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,                 # CHAT GPT에 각각 계산 방식 물어보기 #
    eval_metric = 'mlogloss',       # 다중 분류 : mlogloss, merror
                                    # 이진 분류 : logloss, error
                                    # 회귀      : rmse, mae, rmsle
                                    # 2.1.1 버전 이후로 fit 에서 모델로 위치 변경
    early_stopping_rounds=10,
    random_state=RS)
    
    select_model.fit(select_x_trn, y_trn,
                     eval_set = [(select_x_tst,y_tst)],
                     verbose = False)
                
    score = select_model.score(select_x_tst,y_tst)
    print(f'Threshold = {i:.3f} / n = {select_x_trn.shape[1]:2d} / R2 = {score*100:.3f}%')
    
# ORIG_SCR : 1.0
# Quantile : 0.25
# DROP_SCR : 1.0
# PLUS_SCR : 1.0
# DROP_COL : ['total_phenols', 'nonflavanoid_phenols', 'proanthocyanins', 'od280/od315_of_diluted_wines']

# R2 : 1.0
# [0.01622403 0.06677894 0.05647606 0.04464573 0.07179123 0.04102067
#  0.10167695 0.05011638 0.00125416 0.1748918  0.04591461 0.14518543
#  0.18402405]

# Threshold = 0.001 / n = 13 / R2 = 100.000%
# Threshold = 0.016 / n = 12 / R2 = 100.000%
# Threshold = 0.041 / n = 11 / R2 = 94.444%
# Threshold = 0.045 / n = 10 / R2 = 94.444%
# Threshold = 0.046 / n =  9 / R2 = 94.444%
# Threshold = 0.050 / n =  8 / R2 = 94.444%
# Threshold = 0.056 / n =  7 / R2 = 94.444%
# Threshold = 0.067 / n =  6 / R2 = 94.444%
# Threshold = 0.072 / n =  5 / R2 = 94.444%
# Threshold = 0.102 / n =  4 / R2 = 100.000%
# Threshold = 0.145 / n =  3 / R2 = 100.000%
# Threshold = 0.175 / n =  2 / R2 = 88.889%
# Threshold = 0.184 / n =  1 / R2 = 72.222%