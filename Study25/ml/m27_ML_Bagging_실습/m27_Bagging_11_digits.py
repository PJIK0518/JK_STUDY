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

warnings.filterwarnings('ignore')

#2. 모델
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor, RandomForestClassifier

# model = DecisionTreeClassifier()
# model = BaggingClassifier(DecisionTreeClassifier(),
#                          n_estimators= 100,
#                          n_jobs=-1,
#                          random_state=RS,
#                          bootstrap=False
#                          )
model = RandomForestClassifier(random_state=RS)

#3. 훈련
model.fit(x_trn, y_trn)

#4. 평가 예측
rslt = model.score(x_tst, y_tst)

print('점수:', rslt)

# DecisionTreeRegressor  점수: 0.8833333333333333

# BaggingRegressor + DTR 점수: 0.9722222222222222 >> 직접 DTR을 Bagging 시킨 모델 : 점수 또이또이
            # bootstrap = True
            
# BaggingRegressor + DTR 점수: 0.9
            # bootstrap = False
            # Sample 데이터 중복 허용
            
# RandomForestRegressor  점수: 0.9722222222222222 >> DTR이 Bagging 되어있는 모델  : 점수 또이또이