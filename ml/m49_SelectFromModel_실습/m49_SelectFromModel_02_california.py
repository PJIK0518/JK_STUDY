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

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.8,
    random_state=RS,  
)
# print(x.shape, y.shape) (20640, 8) (20640,)
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
    random_state=RS)
    
    select_model.fit(select_x_trn, y_trn,
                     eval_set = [(select_x_tst,y_tst)],
                     verbose = False)
                
    score = select_model.score(select_x_tst,y_tst)
    print(f'Threshold = {i:.3f} / n = {select_x_trn.shape[1]:2d} / R2 = {score*100:.3f}%')

# [0.46729922 0.06602691 0.04730553 0.02748645 0.02454475 0.1568825
#  0.10187978 0.10857485]
# ORIG_SCR : 0.8362797172573576
# (14447, 1) (6193, 1)
# Quantile : 0.25
# PLUS_SCR : 0.999998024743793

# R2 : 0.8203498635673543
# [0.40068996 0.06617637 0.0626711  0.04642041 0.04546    0.14676294
#  0.10882124 0.12299796]
# Threshold = 0.045 / n =  8 / R2 = 82.035%
# Threshold = 0.046 / n =  7 / R2 = 81.289%
# Threshold = 0.063 / n =  6 / R2 = 81.596%
# Threshold = 0.066 / n =  5 / R2 = 81.418%
# Threshold = 0.109 / n =  4 / R2 = 80.636%
# Threshold = 0.123 / n =  3 / R2 = 70.512%
# Threshold = 0.147 / n =  2 / R2 = 59.358%
# Threshold = 0.401 / n =  1 / R2 = 48.753%