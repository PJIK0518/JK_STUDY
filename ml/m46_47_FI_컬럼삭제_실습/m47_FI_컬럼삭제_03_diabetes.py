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

###############################
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
from xgboost import XGBRegressor, XGBClassifier
import pandas as pd

Q = 25
#2 모델구성
model = XGBRegressor(random_state=42)

model.fit(x_trn, y_trn)
print('ORIG_SCR :', model.score(x_tst, y_tst))
                                                                     
CPT = np.percentile(model.feature_importances_, Q)

COL_name = []

for i, FI in enumerate(model.feature_importances_):
    if FI <= CPT:
        COL_name.append(DS.feature_names[i])
    else:
        continue

x = pd.DataFrame(x, columns=DS.feature_names)
x = x.drop(columns=COL_name)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42,
    # stratify=y,
    )

model.fit(x_trn, y_trn)

score = model.score(x_trn, y_trn)
print('Quantile :', Q/100)
print('DROP_SCR :', score)

# ORIG_SCR : -0.020787322971324906
# Quantile : 0.25
# DROP_SCR : 0.9999898038583587