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

#####################################
## Scaler
def Scaler(SC, a, b):
    SC.fit(a)
    a_scaled = pd.DataFrame(SC.transform(a), columns=a.columns, index=a.index)
    b_scaled = pd.DataFrame(SC.transform(b), columns=a.columns, index=b.index)
    return a_scaled, b_scaled

#####################################
## 증폭 : class_weight
classes = np.unique(y)
CW = compute_class_weight(class_weight='balanced', classes=classes, y=y)

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
import warnings
import numpy as np
import random
import time

RS =44
#2. 모델
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor, RandomForestClassifier

# model = DecisionTreeClassifier()
model = BaggingClassifier(DecisionTreeClassifier(),
                         n_estimators= 100,
                         n_jobs=-1,
                         random_state=RS,
                         bootstrap=False
                         )
# model = RandomForestClassifier(random_state=RS)

#3. 훈련
model.fit(x_trn, y_trn)

#4. 평가 예측
rslt = model.score(x_tst, y_tst)

print('점수:', rslt)

# DecisionTreeRegressor  점수: 0.8343

# BaggingRegressor + DTR 점수: 점수: 0.9017 >> 직접 DTR을 Bagging 시킨 모델 : 점수 또이또이
            # bootstrap = True
            
# BaggingRegressor + DTR 점수: 
            # bootstrap = False
            # Sample 데이터 중복 허용
            
# RandomForestRegressor  점수:  >> DTR이 Bagging 되어있는 모델  : 점수 또이또이   

# {'target': 0.023627735113659787,
#  'params': {'colsample_bylevel': 0.5, 
#             'colsample_bytree': 0.5, 
#             'gamma': 4.501043639845214,
#             'learning_rate': 0.1, 
#             'max_bin': 200.50477676693572,
#             'max_depth': 10.0,
#             'min_child_samples': 200.0, 
#             'min_child_weight': 50.0,
#             'n_estimator': 320.0270780438317, 
#             'num_leaves': 24.0, 
#             'reg_alpha': 1.142586573446205,
#             'reg_lambda': 0.0, 
#             'subsample': 0.5966838594094834}}