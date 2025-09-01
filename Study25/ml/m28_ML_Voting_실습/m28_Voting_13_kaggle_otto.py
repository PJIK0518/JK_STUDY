# m10_13.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#           >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#           >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#           >> 데이터의 소실 없이 훈련 가능
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

#1. 데이터
path = './Study25/_data/kaggle/otto/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = trn_csv.drop(['target'], axis=1)
y = trn_csv['target']

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)

#####################################
## Scaler
def Scaler(SC, a, b):
    SC.fit(a)
    a_scaled = pd.DataFrame(SC.transform(a), columns=a.columns, index=a.index)
    b_scaled = pd.DataFrame(SC.transform(b), columns=a.columns, index=b.index)
    return a_scaled, b_scaled

x_trn, tst_csv = Scaler(MaxAbsScaler(), x_trn, tst_csv)

#####################################
## 증폭 : class_weight
classes = np.unique(y)
CW = compute_class_weight(class_weight='balanced', classes=classes, y=y)

#####################################
## Label_y
LE = LabelEncoder()
LE.fit(y_trn)
y_trn = LE.transform(y_trn)
y_tst = LE.transform(y_tst)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
import warnings
import numpy as np
import random
import time


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
# vote = 'soft'
model = VotingClassifier(
    estimators=[('XGB', XGB),
                ('LGB', LGB),
                ('CAT', CAT)],
    voting= vote,
)

#3. 훈련
model.fit(x_trn, y_trn)

#4. 평가 예측
rslt = model.score(x_tst, y_tst)

### hard ###
# # 개별 예측 수동 조합
# XGB.fit(x_trn, y_trn)
# LGB.fit(x_trn, y_trn)
# CAT.fit(x_trn, y_trn)

# # 예측 결과 (전부 1차원으로 평탄화)
# preds = np.array([
#     XGB.predict(x_tst),
#     LGB.predict(x_tst),
#     CAT.predict(x_tst).ravel()
# ])

# # 하드보팅 → 다수결
# from scipy.stats import mode
# final_pred = mode(preds, axis=0).mode[0]

# rslt = accuracy_score(y_tst, final_pred)

# print(f'{vote}:', rslt)

# DecisionTreeRegressor  점수: 

# BaggingRegressor + DTR 점수:  >> 직접 DTR을 Bagging 시킨 모델 : 점수 또이또이
            # bootstrap = True
            
# BaggingRegressor + DTR 점수: 
            # bootstrap = False
            # Sample 데이터 중복 허용
            
# RandomForestRegressor  점수:  >> DTR이 Bagging 되어있는 모델  : 점수 또이또이   

# {'target': 0.3361253998373447,
#  'params': {'colsample_bylevel': 0.6150592313147155,
#             'colsample_bytree': 1.0,
#             'gamma': 5.0, 'learning_rate': 0.1,
#             'max_bin': 343.4072919726811,
#             'max_depth': 3.0,
#             'min_child_samples': 43.192163305392114,
#             'min_child_weight': 1.0,
#             'n_estimator': 456.0235062465993,
#             'num_leaves': 33.260173446440554,
#             'reg_alpha': 8.32230482255393,
#             'reg_lambda': 25.320411088502198,
#             'subsample': 1.0}}