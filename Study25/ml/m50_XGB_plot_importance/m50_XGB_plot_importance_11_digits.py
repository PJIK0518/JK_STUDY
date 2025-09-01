# m10_11.copy

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
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier

#1. 데이터
DS = load_digits()

x = DS.data
y = DS.target

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

from xgboost.plotting import plot_importance
import matplotlib.pyplot as plt
plot_importance(model)
plt.show()
    
# ORIG_SCR : 0.8117751479148865
# Quantile : 0.25
# DROP_SCR : 0.9999889135360718
# PLUS_SCR : 1.0
# # DROP_COL : ['pixel_0_0', 'pixel_1_0', 'pixel_1_1',
#             'pixel_1_6', 'pixel_1_7', 'pixel_2_1',
#             'pixel_2_7', 'pixel_3_0', 'pixel_3_7',
#             'pixel_4_0', 'pixel_4_7', 'pixel_5_0',
#             'pixel_5_7', 'pixel_6_0', 'pixel_6_7', 'pixel_7_0']

""" R2 : 0.9722222222222222
Threshold = 0.000 / n = 64 / R2 = 97.222%
Threshold = 0.000 / n = 64 / R2 = 97.222%
Threshold = 0.000 / n = 64 / R2 = 97.222%
Threshold = 0.000 / n = 64 / R2 = 97.222%
Threshold = 0.000 / n = 64 / R2 = 97.222%
Threshold = 0.000 / n = 59 / R2 = 97.222%
Threshold = 0.000 / n = 58 / R2 = 97.222%
Threshold = 0.000 / n = 57 / R2 = 97.222%
Threshold = 0.000 / n = 56 / R2 = 97.222%
Threshold = 0.001 / n = 55 / R2 = 97.222%
Threshold = 0.001 / n = 54 / R2 = 97.222%
Threshold = 0.002 / n = 53 / R2 = 97.778%
Threshold = 0.003 / n = 52 / R2 = 97.222%
Threshold = 0.004 / n = 51 / R2 = 97.222%
Threshold = 0.004 / n = 50 / R2 = 98.333%
Threshold = 0.004 / n = 49 / R2 = 97.222%
Threshold = 0.004 / n = 48 / R2 = 97.222%
Threshold = 0.005 / n = 47 / R2 = 98.333%
Threshold = 0.005 / n = 46 / R2 = 97.222%
Threshold = 0.005 / n = 45 / R2 = 97.778%
Threshold = 0.005 / n = 44 / R2 = 97.222%
Threshold = 0.006 / n = 43 / R2 = 98.333%
Threshold = 0.006 / n = 42 / R2 = 97.778%
Threshold = 0.006 / n = 41 / R2 = 97.778%
Threshold = 0.006 / n = 40 / R2 = 97.222%
Threshold = 0.007 / n = 39 / R2 = 97.222%
Threshold = 0.007 / n = 38 / R2 = 98.333%
Threshold = 0.007 / n = 37 / R2 = 97.222%
Threshold = 0.008 / n = 36 / R2 = 96.667%
Threshold = 0.008 / n = 35 / R2 = 97.222%
Threshold = 0.009 / n = 34 / R2 = 96.667%
Threshold = 0.009 / n = 33 / R2 = 97.222%
Threshold = 0.010 / n = 32 / R2 = 96.667%
Threshold = 0.011 / n = 31 / R2 = 96.667%
Threshold = 0.011 / n = 30 / R2 = 96.111%
Threshold = 0.012 / n = 29 / R2 = 97.222%
Threshold = 0.012 / n = 28 / R2 = 96.111%
Threshold = 0.013 / n = 27 / R2 = 95.556%
Threshold = 0.015 / n = 26 / R2 = 96.111%
Threshold = 0.015 / n = 25 / R2 = 95.000%
Threshold = 0.016 / n = 24 / R2 = 93.889%
Threshold = 0.017 / n = 23 / R2 = 97.222%
Threshold = 0.017 / n = 22 / R2 = 96.667%
Threshold = 0.019 / n = 21 / R2 = 95.000%
Threshold = 0.019 / n = 20 / R2 = 94.444%
Threshold = 0.020 / n = 19 / R2 = 95.000%
Threshold = 0.020 / n = 18 / R2 = 95.556%
Threshold = 0.023 / n = 17 / R2 = 95.556%
Threshold = 0.023 / n = 16 / R2 = 95.556%
Threshold = 0.024 / n = 15 / R2 = 93.889%
Threshold = 0.024 / n = 14 / R2 = 94.444%
Threshold = 0.025 / n = 13 / R2 = 93.333%
Threshold = 0.027 / n = 12 / R2 = 93.889%
Threshold = 0.034 / n = 11 / R2 = 91.111%
Threshold = 0.035 / n = 10 / R2 = 88.889%
Threshold = 0.036 / n =  9 / R2 = 88.333%
Threshold = 0.037 / n =  8 / R2 = 86.667%
Threshold = 0.045 / n =  7 / R2 = 85.000%
Threshold = 0.046 / n =  6 / R2 = 80.000%
Threshold = 0.047 / n =  5 / R2 = 67.222%
Threshold = 0.049 / n =  4 / R2 = 65.000%
Threshold = 0.049 / n =  3 / R2 = 54.444%
Threshold = 0.060 / n =  2 / R2 = 38.333%
Threshold = 0.063 / n =  1 / R2 = 23.889% """