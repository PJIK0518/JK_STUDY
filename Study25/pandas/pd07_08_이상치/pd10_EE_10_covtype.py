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
import pandas as pdc

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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
def outlier(data):
    out = []
    up = []
    low = []
    for i in range(data.shape[1]):
        col = data[:, i]
        Q1, Q3 = np.percentile(col, [25, 75])
        
        IQR = Q3 - Q1
        print('IQR :', IQR)
        
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        
        out_i = np.where((col > upper_bound) | (col < lower_bound))[0]
        out.append(out_i)
        up.append(upper_bound)
        low.append(lower_bound)
    return out, up, low

OUT, UP, LOW = outlier(x)

print(len(OUT[0])) # [array([ 0, 12]), array([6])]
# print(UP)  # [19.0, 1200.0]
# print(LOW) # [-5.0, -400.0]

# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(6,9, figsize=(15, 9))
# axs = axs.flatten()

# for i in range(x.shape[1]):
#     axs[i].boxplot(x[:,i])
#     axs[i].axhline(UP[i], color = 'red', label = 'upper_bound')
#     axs[i].axhline(LOW[i], color = 'red', label = 'lower_bound')
#     axs[i].set_title(f"Column {i}")
    
# plt.tight_layout()
# plt.show()
# exit()

from sklearn.preprocessing import RobustScaler
RSC = RobustScaler()
col = [range(x.shape[1])]

for i in col:
    x_trn_col = x_trn[:, i].reshape(-1, 1)
    x_tst_col = x_tst[:, i].reshape(-1, 1)
    
    RSC.fit(x_trn_col)
    x_trn[:, i] = RSC.transform(x_trn_col).reshape(-1, 54)
    x_tst[:, i] = RSC.transform(x_tst_col).reshape(-1, 54)


#2 모델구성
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('R2 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

score1 = r2_score(y_tst, y_prd)
print('# Outlier 처리')
print('R2 :',score1)
# Outlier 처리
# R2 : 0.3214405991405006
# PF R2 : 0.4816085978219853
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9722901104953358
# ACC : 0.9722901104953358

# F1S : 0.9432327280233942

# ORIG_SCR : 0.8671302192695605
# Quantile : 0.25
# DROP_SCR : 0.8840863715491213
# ORIG_SCR : 0.8671302192695605
# DROP_COL : ['Slope', 'Soil_Type_6', 'Soil_Type_7',
#             'Soil_Type_8', 'Soil_Type_14', 'Soil_Type_15',
#             'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_20',
#             'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26',
#             'Soil_Type_27', 'Soil_Type_35']

""" R2 : 0.9174727203882826
Threshold = 0.002 / n = 54 / R2 = 91.747%
Threshold = 0.003 / n = 53 / R2 = 91.747%
Threshold = 0.004 / n = 52 / R2 = 91.780%
Threshold = 0.004 / n = 51 / R2 = 91.766%
Threshold = 0.004 / n = 50 / R2 = 91.871%
Threshold = 0.006 / n = 49 / R2 = 91.813%
Threshold = 0.006 / n = 48 / R2 = 91.937%
Threshold = 0.006 / n = 47 / R2 = 91.897%
Threshold = 0.006 / n = 46 / R2 = 91.777%
Threshold = 0.006 / n = 45 / R2 = 91.983%
Threshold = 0.007 / n = 44 / R2 = 91.835%
Threshold = 0.007 / n = 43 / R2 = 91.895%
Threshold = 0.008 / n = 42 / R2 = 91.959%
Threshold = 0.008 / n = 41 / R2 = 91.677%
Threshold = 0.008 / n = 40 / R2 = 90.861%
Threshold = 0.009 / n = 39 / R2 = 90.851%
Threshold = 0.010 / n = 38 / R2 = 89.933%
Threshold = 0.010 / n = 37 / R2 = 89.913%
Threshold = 0.011 / n = 36 / R2 = 89.950%
Threshold = 0.012 / n = 35 / R2 = 90.135%
Threshold = 0.012 / n = 34 / R2 = 90.181%
Threshold = 0.012 / n = 33 / R2 = 87.923%
Threshold = 0.012 / n = 32 / R2 = 87.851%
Threshold = 0.012 / n = 31 / R2 = 80.130%
Threshold = 0.012 / n = 30 / R2 = 80.207%
Threshold = 0.013 / n = 29 / R2 = 80.133%
Threshold = 0.013 / n = 28 / R2 = 72.789%
Threshold = 0.014 / n = 27 / R2 = 72.629%
Threshold = 0.015 / n = 26 / R2 = 72.553%
Threshold = 0.015 / n = 25 / R2 = 72.600%
Threshold = 0.016 / n = 24 / R2 = 72.603%
Threshold = 0.016 / n = 23 / R2 = 72.564%
Threshold = 0.018 / n = 22 / R2 = 72.450%
Threshold = 0.018 / n = 21 / R2 = 72.438%
Threshold = 0.020 / n = 20 / R2 = 72.247%
Threshold = 0.021 / n = 19 / R2 = 72.242%
Threshold = 0.021 / n = 18 / R2 = 71.836%
Threshold = 0.022 / n = 17 / R2 = 71.820%
Threshold = 0.022 / n = 16 / R2 = 71.731%
Threshold = 0.023 / n = 15 / R2 = 71.734%
Threshold = 0.024 / n = 14 / R2 = 71.650%
Threshold = 0.026 / n = 13 / R2 = 71.622%
Threshold = 0.027 / n = 12 / R2 = 71.593%
Threshold = 0.028 / n = 11 / R2 = 71.509%
Threshold = 0.031 / n = 10 / R2 = 70.803%
Threshold = 0.031 / n =  9 / R2 = 70.605%
Threshold = 0.034 / n =  8 / R2 = 70.189%
Threshold = 0.034 / n =  7 / R2 = 70.003%
Threshold = 0.037 / n =  6 / R2 = 69.385%
Threshold = 0.041 / n =  5 / R2 = 69.101%
Threshold = 0.048 / n =  4 / R2 = 68.292%
Threshold = 0.052 / n =  3 / R2 = 68.283%
Threshold = 0.061 / n =  2 / R2 = 67.721%
Threshold = 0.064 / n =  1 / R2 = 48.584% """

