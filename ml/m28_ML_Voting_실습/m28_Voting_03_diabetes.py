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

#2. 모델
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

XGB = XGBRegressor()
LGB = LGBMRegressor()
CAT = CatBoostRegressor()

# vote = 'hard'
vote = 'soft'
model = VotingRegressor(
    estimators=[('XGB', XGB),
                ('LGB', LGB),
                ('CAT', CAT)],
)

#3. 훈련
model.fit(x_trn, y_trn)

#4. 평가 예측
rslt = model.score(x_tst, y_tst)

print(f'{vote}:', rslt)
# soft: -0.174884585353128

# DecisionTreeRegressor  점수: -0.00016865845987235062

# BaggingRegressor + DTR 점수: -0.2967264752106398 >> 직접 DTR을 Bagging 시킨 모델 : 점수 또이또이
            # bootstrap = True
            
# BaggingRegressor + DTR 점수: -0.035503232825399245
            # bootstrap = False
            # Sample 데이터 중복 허용
            
# RandomForestRegressor  점수: -0.2504551618231017 >> DTR이 Bagging 되어있는 모델  : 점수 또이또이