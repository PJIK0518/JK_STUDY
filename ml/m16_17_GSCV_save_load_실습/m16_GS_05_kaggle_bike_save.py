# m10_05.copy

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

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

#1. 데이터

path = './Study25/_data/kaggle/bike/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test_new_0527_1.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = trn_csv.drop(['count'], axis = 1)
y = trn_csv['count']

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
NS = 5

KF = KFold(n_splits= NS,
                     shuffle=True,
                     random_state=RS)

PM = [
    {'n_estimators': [100,500], 'max_depth': [6,10, 12] ,'learning_rate' : [0.1, 0.01, 0.001]}, # 18
    {'max _depth': [6,8,10,12], 'learning rate' : [0.1, 0.01, 0.001]},                          # 12
    {'min_child weight':[2,3,5,10], 'learning_rate' : [0.1, 0.01,0.001]},                       # 12
]

#2. 모델
model = GridSearchCV(XGBRegressor(),       
                     PM,           
                     cv = KF,     
                     verbose=1, 
                     refit=True,  
                     n_jobs=-1,   
                                 
)

#3. 훈련
S = time.time()
model.fit(x_trn, y_trn)
print('최적 매개변수 :', model.best_estimator_)
print('최적 파라미터 :', model.best_params_)

#4. 평가 예측
print('훈련 최고점수 :', model.best_score_)
print('최고 성능평가 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)
print('실제 모델성능 :', r2_score(y_tst, y_prd))

y_prd_best = model.best_estimator_.predict(x_tst)
print('최고 모델성능 :', r2_score(y_tst, y_prd_best))

saveNum = '05'
path = './Study25/_save/m16_GS_save/'
import joblib
print('훈련 소요시간 :', time.time() - S)
joblib.dump(model.best_estimator_, path + f'm16_best_model_{saveNum}.joblib')
print(saveNum, '저장완료')