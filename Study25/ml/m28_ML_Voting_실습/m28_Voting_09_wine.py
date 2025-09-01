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

warnings.filterwarnings('ignore')

#2. 모델
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

XGB = XGBClassifier()
LGB = LGBMClassifier(
    verbosity = -1
)
CAT = CatBoostClassifier(
        verbose=0,
)

vote = 'hard'
# # vote = 'soft'
# model = VotingClassifier(
#     estimators=[('XGB', XGB),
#                 ('LGB', LGB),
#                 ('CAT', CAT)],
#     voting= vote,
# )

# #3. 훈련
# model.fit(x_trn, y_trn)

# #4. 평가 예측
# rslt = model.score(x_tst, y_tst)

### hard ###
# 개별 예측 수동 조합
XGB.fit(x_trn, y_trn)
LGB.fit(x_trn, y_trn)
CAT.fit(x_trn, y_trn)

# 예측 결과 (전부 1차원으로 평탄화)
preds = np.array([
    XGB.predict(x_tst),
    LGB.predict(x_tst),
    CAT.predict(x_tst).ravel()
])

# 하드보팅 → 다수결
from scipy.stats import mode
final_pred = mode(preds, axis=0).mode[0]

rslt = accuracy_score(y_tst, final_pred)

print(f'{vote}:', rslt)
# soft: 1.0
# hard: 1.0