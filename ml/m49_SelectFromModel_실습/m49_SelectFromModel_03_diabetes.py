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

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb

#2 모델구성
model = XGBRegressor(
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
    random_state=seed
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
    
    select_model = XGBRegressor(
    n_estimators = 200,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,                 # CHAT GPT에 각각 계산 방식 물어보기 #
    eval_metric = 'rmse',     # 다중 분류 : mlogloss, merror
                                    # 이진 분류 : logloss, error
                                    # 회귀      : rmse, mae, rmsle
                                    # 2.1.1 버전 이후로 fit 에서 모델로 위치 변경
    early_stopping_rounds=10,
    random_state=seed)
    
    select_model.fit(select_x_trn, y_trn,
                     eval_set = [(select_x_tst,y_tst)],
                     verbose = False)
                
    score = select_model.score(select_x_tst,y_tst)
    print(f'Threshold = {i:.3f} / n = {select_x_trn.shape[1]:2d} / R2 = {score*100:.3f}%')

# [0.03538059 0.05027447 0.17167118 0.08811221 0.0732259 
#  0.06742777 0.07867986 0.171908   0.163297   0.10002296]
# Threshold = 0.035 / n = 10 / R2 = 34.272%
# Threshold = 0.050 / n =  9 / R2 = 37.315%
# Threshold = 0.067 / n =  8 / R2 = 34.110%
# Threshold = 0.073 / n =  7 / R2 = 32.583%
# Threshold = 0.079 / n =  6 / R2 = 43.030%
# Threshold = 0.088 / n =  5 / R2 = 40.817%
# Threshold = 0.100 / n =  4 / R2 = 34.464%
# Threshold = 0.163 / n =  3 / R2 = 31.007%
# Threshold = 0.172 / n =  2 / R2 = 23.018%
# Threshold = 0.172 / n =  1 / R2 = 13.971%