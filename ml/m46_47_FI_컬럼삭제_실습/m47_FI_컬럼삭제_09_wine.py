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

from xgboost import XGBRegressor, XGBClassifier
import pandas as pd

Q = 25
#2 모델구성
model = XGBClassifier(random_state=42)

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
    stratify=y,
    )

model.fit(x_trn, y_trn)

score = model.score(x_trn, y_trn)
print('Quantile :', Q/100)
print('DROP_SCR :', score)

# ORIG_SCR : 1.0
# Quantile : 0.25
# DROP_SCR : 1.0

# n_components=  1 | acc=0.6111
# n_components= 12 | acc=0.8333
# n_components= 13 | acc=0.8333
# n_components= 13 | acc=0.8333    
    
# CatBoostClassifier | train pred shape: (160, 1), test pred shape: (18, 1)
# XGBClassifier | train pred shape: (160,), test pred shape: (18,)
# RandomForestClassifier | train pred shape: (160,), test pred shape: (18,)
# LGBMClassifier | train pred shape: (160,), test pred shape: (18,)
# Stacking score : 1.0

# soft: 1.0
# hard: 1.0