# m10_12.copy

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

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


#1. 데이터
path = './Study25/_data/kaggle/santander/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

x = trn_csv.drop(['target'], axis=1)
y = trn_csv['target']

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)
from sklearn.decomposition import PCA
import time
#####################################
## Scaler
def Scaler(SC, a, b):
    SC.fit(a)
    a_scaled = SC.transform(a)
    b_scaled = SC.transform(b)
    return a_scaled, b_scaled

x_trn, x_tst = Scaler(RobustScaler(), x_trn, x_tst)
x_trn, x_tst = Scaler(StandardScaler(), x_trn, x_tst)

RS = 777

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
# print(model.feature_importances_)


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
    # eval_metric = 'mlogloss',     # 다중 분류 : mlogloss, merror
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

# ORIG_SCR : 0.9161
# Quantile : 0.25
# DROP_SCR : 0.9555357142857143
# PLUS_SCR : 0.9556071428571429

""" R2 : 0.91055 
Threshold = 0.002 / n = 200 / R2 = 91.055%
Threshold = 0.002 / n = 199 / R2 = 91.150%
Threshold = 0.003 / n = 198 / R2 = 91.180%
Threshold = 0.003 / n = 197 / R2 = 91.080%
Threshold = 0.003 / n = 196 / R2 = 91.240%
Threshold = 0.003 / n = 195 / R2 = 91.230%
Threshold = 0.003 / n = 194 / R2 = 91.210%
Threshold = 0.003 / n = 193 / R2 = 91.225%
Threshold = 0.003 / n = 192 / R2 = 91.165%
Threshold = 0.003 / n = 191 / R2 = 91.310%
Threshold = 0.003 / n = 190 / R2 = 91.150%
Threshold = 0.003 / n = 189 / R2 = 91.090%
Threshold = 0.003 / n = 188 / R2 = 91.130%
Threshold = 0.003 / n = 187 / R2 = 91.050%
Threshold = 0.003 / n = 186 / R2 = 91.180%
Threshold = 0.003 / n = 185 / R2 = 91.245%
Threshold = 0.003 / n = 184 / R2 = 91.250%
Threshold = 0.003 / n = 183 / R2 = 91.320%
Threshold = 0.003 / n = 182 / R2 = 91.355%
Threshold = 0.003 / n = 181 / R2 = 91.240%
Threshold = 0.003 / n = 180 / R2 = 91.085%
Threshold = 0.003 / n = 179 / R2 = 91.155%
Threshold = 0.003 / n = 178 / R2 = 91.205%
Threshold = 0.003 / n = 177 / R2 = 91.295%
Threshold = 0.003 / n = 176 / R2 = 91.220%
Threshold = 0.003 / n = 175 / R2 = 91.185%
Threshold = 0.003 / n = 174 / R2 = 91.155%
Threshold = 0.003 / n = 173 / R2 = 91.315%
Threshold = 0.003 / n = 172 / R2 = 91.220%
Threshold = 0.003 / n = 171 / R2 = 91.310%
Threshold = 0.003 / n = 170 / R2 = 91.305%
Threshold = 0.003 / n = 169 / R2 = 91.350%
Threshold = 0.003 / n = 168 / R2 = 91.090%
Threshold = 0.003 / n = 167 / R2 = 91.215%
Threshold = 0.003 / n = 166 / R2 = 91.255%
Threshold = 0.003 / n = 165 / R2 = 91.220%
Threshold = 0.003 / n = 164 / R2 = 91.190%
Threshold = 0.003 / n = 163 / R2 = 91.050%
Threshold = 0.003 / n = 162 / R2 = 91.150%
Threshold = 0.003 / n = 161 / R2 = 91.275%
Threshold = 0.003 / n = 160 / R2 = 91.335%
Threshold = 0.003 / n = 159 / R2 = 91.270%
Threshold = 0.003 / n = 158 / R2 = 91.150%
Threshold = 0.003 / n = 157 / R2 = 91.260%
Threshold = 0.003 / n = 156 / R2 = 91.305%
Threshold = 0.003 / n = 155 / R2 = 91.060%
Threshold = 0.003 / n = 154 / R2 = 91.010%
Threshold = 0.003 / n = 153 / R2 = 91.115%
Threshold = 0.004 / n = 152 / R2 = 91.010%
Threshold = 0.004 / n = 151 / R2 = 91.295%
Threshold = 0.004 / n = 150 / R2 = 91.130%
Threshold = 0.004 / n = 149 / R2 = 91.280%
Threshold = 0.004 / n = 148 / R2 = 91.185%
Threshold = 0.004 / n = 147 / R2 = 91.460%
Threshold = 0.004 / n = 146 / R2 = 91.300%
Threshold = 0.004 / n = 145 / R2 = 91.280%
Threshold = 0.004 / n = 144 / R2 = 91.220%
Threshold = 0.004 / n = 143 / R2 = 91.295%
Threshold = 0.004 / n = 142 / R2 = 91.270%
Threshold = 0.004 / n = 141 / R2 = 91.305%
Threshold = 0.004 / n = 140 / R2 = 91.385%
Threshold = 0.004 / n = 139 / R2 = 91.115%
Threshold = 0.004 / n = 138 / R2 = 91.225%
Threshold = 0.004 / n = 137 / R2 = 91.145%
Threshold = 0.004 / n = 136 / R2 = 91.190%
Threshold = 0.004 / n = 135 / R2 = 91.000%
Threshold = 0.004 / n = 134 / R2 = 91.195%
Threshold = 0.004 / n = 133 / R2 = 91.140%
Threshold = 0.004 / n = 132 / R2 = 91.295%
Threshold = 0.004 / n = 131 / R2 = 91.305%
Threshold = 0.004 / n = 130 / R2 = 91.245%
Threshold = 0.004 / n = 129 / R2 = 91.220%
Threshold = 0.004 / n = 128 / R2 = 91.385%
Threshold = 0.004 / n = 127 / R2 = 91.285%
Threshold = 0.004 / n = 126 / R2 = 91.300%
Threshold = 0.004 / n = 125 / R2 = 91.170%
Threshold = 0.004 / n = 124 / R2 = 91.200%
Threshold = 0.004 / n = 123 / R2 = 91.125%
Threshold = 0.004 / n = 122 / R2 = 91.290%
Threshold = 0.004 / n = 121 / R2 = 91.280%
Threshold = 0.004 / n = 120 / R2 = 91.200%
Threshold = 0.004 / n = 119 / R2 = 91.340%
Threshold = 0.004 / n = 118 / R2 = 91.225%
Threshold = 0.004 / n = 117 / R2 = 90.960%
Threshold = 0.004 / n = 116 / R2 = 91.145%
Threshold = 0.004 / n = 115 / R2 = 91.280%
Threshold = 0.004 / n = 114 / R2 = 91.240%
Threshold = 0.004 / n = 113 / R2 = 91.160%
Threshold = 0.004 / n = 112 / R2 = 91.345%
Threshold = 0.004 / n = 111 / R2 = 91.090%
Threshold = 0.004 / n = 110 / R2 = 91.160%
Threshold = 0.004 / n = 109 / R2 = 91.085%
Threshold = 0.004 / n = 108 / R2 = 91.265%
Threshold = 0.004 / n = 107 / R2 = 91.170%
Threshold = 0.004 / n = 106 / R2 = 91.025%
Threshold = 0.004 / n = 105 / R2 = 91.115%
Threshold = 0.004 / n = 104 / R2 = 91.185%
Threshold = 0.004 / n = 103 / R2 = 91.195%
Threshold = 0.004 / n = 102 / R2 = 91.085%
Threshold = 0.005 / n = 101 / R2 = 91.000%
Threshold = 0.005 / n = 100 / R2 = 90.990%
Threshold = 0.005 / n = 99 / R2 = 91.290%
Threshold = 0.005 / n = 98 / R2 = 91.200%
Threshold = 0.005 / n = 97 / R2 = 91.015%
Threshold = 0.005 / n = 96 / R2 = 91.065%
Threshold = 0.005 / n = 95 / R2 = 91.225%
Threshold = 0.005 / n = 94 / R2 = 90.945%
Threshold = 0.005 / n = 93 / R2 = 91.115%
Threshold = 0.005 / n = 92 / R2 = 91.025%
Threshold = 0.005 / n = 91 / R2 = 91.015%
Threshold = 0.005 / n = 90 / R2 = 91.125%
Threshold = 0.005 / n = 89 / R2 = 91.095%
Threshold = 0.005 / n = 88 / R2 = 91.140%
Threshold = 0.005 / n = 87 / R2 = 91.075%
Threshold = 0.005 / n = 86 / R2 = 91.000%
Threshold = 0.005 / n = 85 / R2 = 91.195%
Threshold = 0.005 / n = 84 / R2 = 91.110%
Threshold = 0.005 / n = 83 / R2 = 91.130%
Threshold = 0.005 / n = 82 / R2 = 90.920%
Threshold = 0.005 / n = 81 / R2 = 91.080%
Threshold = 0.005 / n = 80 / R2 = 91.040%
Threshold = 0.005 / n = 79 / R2 = 91.190%
Threshold = 0.005 / n = 78 / R2 = 91.035%
Threshold = 0.005 / n = 77 / R2 = 91.110%
Threshold = 0.005 / n = 76 / R2 = 91.035%
Threshold = 0.005 / n = 75 / R2 = 90.895%
Threshold = 0.005 / n = 74 / R2 = 90.920%
Threshold = 0.005 / n = 73 / R2 = 91.105%
Threshold = 0.005 / n = 72 / R2 = 90.895%
Threshold = 0.005 / n = 71 / R2 = 91.125%
Threshold = 0.005 / n = 70 / R2 = 90.900%
Threshold = 0.005 / n = 69 / R2 = 90.960%
Threshold = 0.005 / n = 68 / R2 = 90.895%
Threshold = 0.005 / n = 67 / R2 = 90.955%
Threshold = 0.005 / n = 66 / R2 = 90.985%
Threshold = 0.005 / n = 65 / R2 = 90.855%
Threshold = 0.006 / n = 64 / R2 = 90.900%
Threshold = 0.006 / n = 63 / R2 = 90.790%
Threshold = 0.006 / n = 62 / R2 = 90.880%
Threshold = 0.006 / n = 61 / R2 = 90.895%
Threshold = 0.006 / n = 60 / R2 = 90.730%
Threshold = 0.006 / n = 59 / R2 = 90.920%
Threshold = 0.006 / n = 58 / R2 = 90.690%
Threshold = 0.006 / n = 57 / R2 = 90.710%
Threshold = 0.006 / n = 56 / R2 = 90.700%
Threshold = 0.006 / n = 55 / R2 = 90.755%
Threshold = 0.006 / n = 54 / R2 = 90.765%
Threshold = 0.006 / n = 53 / R2 = 90.680%
Threshold = 0.006 / n = 52 / R2 = 90.545%
Threshold = 0.006 / n = 51 / R2 = 90.725%
Threshold = 0.006 / n = 50 / R2 = 90.755%
Threshold = 0.006 / n = 49 / R2 = 90.600%
Threshold = 0.006 / n = 48 / R2 = 90.655%
Threshold = 0.006 / n = 47 / R2 = 90.550%
Threshold = 0.006 / n = 46 / R2 = 90.610%
Threshold = 0.006 / n = 45 / R2 = 90.615%
Threshold = 0.006 / n = 44 / R2 = 90.600%
Threshold = 0.006 / n = 43 / R2 = 90.625%
Threshold = 0.007 / n = 42 / R2 = 90.455%
Threshold = 0.007 / n = 41 / R2 = 90.495%
Threshold = 0.007 / n = 40 / R2 = 90.540%
Threshold = 0.007 / n = 39 / R2 = 90.400%
Threshold = 0.007 / n = 38 / R2 = 90.400%
Threshold = 0.007 / n = 37 / R2 = 90.515%
Threshold = 0.007 / n = 36 / R2 = 90.325%
Threshold = 0.007 / n = 35 / R2 = 90.335%
Threshold = 0.007 / n = 34 / R2 = 90.440%
Threshold = 0.007 / n = 33 / R2 = 90.485%
Threshold = 0.007 / n = 32 / R2 = 90.265%
Threshold = 0.007 / n = 31 / R2 = 90.290%
Threshold = 0.007 / n = 30 / R2 = 90.165%
Threshold = 0.007 / n = 29 / R2 = 90.370%
Threshold = 0.007 / n = 28 / R2 = 90.350%
Threshold = 0.007 / n = 27 / R2 = 90.080%
Threshold = 0.007 / n = 26 / R2 = 90.070%
Threshold = 0.007 / n = 25 / R2 = 90.250%
Threshold = 0.007 / n = 24 / R2 = 90.120%
Threshold = 0.007 / n = 23 / R2 = 90.075%
Threshold = 0.008 / n = 22 / R2 = 90.175%
Threshold = 0.008 / n = 21 / R2 = 90.165%
Threshold = 0.008 / n = 20 / R2 = 90.095%
Threshold = 0.008 / n = 19 / R2 = 90.070%
Threshold = 0.008 / n = 18 / R2 = 90.100%
Threshold = 0.008 / n = 17 / R2 = 89.925%
Threshold = 0.008 / n = 16 / R2 = 90.090%
Threshold = 0.008 / n = 15 / R2 = 90.165%
Threshold = 0.008 / n = 14 / R2 = 90.030%
Threshold = 0.008 / n = 13 / R2 = 90.010%
Threshold = 0.009 / n = 12 / R2 = 90.090%
Threshold = 0.009 / n = 11 / R2 = 90.020%
Threshold = 0.009 / n = 10 / R2 = 89.980%
Threshold = 0.009 / n =  9 / R2 = 89.930%
Threshold = 0.009 / n =  8 / R2 = 89.965%
Threshold = 0.009 / n =  7 / R2 = 90.055%
Threshold = 0.009 / n =  6 / R2 = 90.005%
Threshold = 0.009 / n =  5 / R2 = 90.005%
Threshold = 0.010 / n =  4 / R2 = 90.035%
Threshold = 0.010 / n =  3 / R2 = 90.035%
Threshold = 0.010 / n =  2 / R2 = 90.035%
Threshold = 0.012 / n =  1 / R2 = 90.035% """