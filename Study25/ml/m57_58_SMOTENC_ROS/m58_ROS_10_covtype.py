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

import time
S= time.time()

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
from imblearn.over_sampling import RandomOverSampler

ROS = RandomOverSampler(random_state=337,
                        sampling_strategy='auto')

x_trn, y_trn = ROS.fit_resample(x_trn, y_trn)
# All auto
# F1S : 0.8905748192420028
# ACC : 89.62686310281917 %
# 94.0 초

# ALL JH_smote
# F1S : 0.887352467513327
# ACC : 89.69398643764414 %
# 190.6 초

# 분할
# F1S : 0.893915536612878
# ACC : 89.986575333035 %
# 70.8 초

# ROS
# F1S : 0.8700701109997423
# ACC : 89.05201197893359 %
# 39.1 초

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

y_prd = model.predict(x_tst)

from sklearn.metrics import f1_score
print('F1S :', f1_score(y_prd,y_tst, average='macro'))
print('ACC :', model.score(x_tst,y_tst)*100,'%')
print(f'{(time.time() - S):.1f}',"초")

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

