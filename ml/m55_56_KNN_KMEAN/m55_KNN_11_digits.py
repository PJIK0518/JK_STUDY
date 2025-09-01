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
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score
model = KNeighborsClassifier(n_neighbors=5)     # k_neighbors랑 같은 Parameter

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('acc :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

score1 = accuracy_score(y_tst, y_prd)
score2 = f1_score(y_tst, y_prd, average='macro')

print('ACC :',score1)
print('F1S :',score2)

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9777777777777777
# ACC : 0.9777777777777777
# F1S : 0.9771982808167019


# 최적 컬럼 : 23 개 
#  ['pixel_0_5' 'pixel_1_4' 'pixel_1_5' 'pixel_2_3' 'pixel_2_5' 'pixel_3_2'
#  'pixel_3_4' 'pixel_4_1' 'pixel_4_4' 'pixel_4_5' 'pixel_4_6' 'pixel_5_2'
#  'pixel_5_3' 'pixel_5_5' 'pixel_6_0' 'pixel_6_3' 'pixel_6_5' 'pixel_6_6'
#  'pixel_7_2' 'pixel_7_3' 'pixel_7_4' 'pixel_7_6' 'pixel_7_7']
# 삭제 컬럼 : 41 개 
#  ['pixel_0_0' 'pixel_0_1' 'pixel_0_2' 'pixel_0_3' 'pixel_0_4' 'pixel_0_6'
#  'pixel_0_7' 'pixel_1_0' 'pixel_1_1' 'pixel_1_2' 'pixel_1_3' 'pixel_1_6'
#  'pixel_1_7' 'pixel_2_0' 'pixel_2_1' 'pixel_2_2' 'pixel_2_4' 'pixel_2_6'
#  'pixel_2_7' 'pixel_3_0' 'pixel_3_1' 'pixel_3_3' 'pixel_3_5' 'pixel_3_6'
#  'pixel_3_7' 'pixel_4_0' 'pixel_4_2' 'pixel_4_3' 'pixel_4_7' 'pixel_5_0'
#  'pixel_5_1' 'pixel_5_4' 'pixel_5_6' 'pixel_5_7' 'pixel_6_1' 'pixel_6_2'
#  'pixel_6_4' 'pixel_6_7' 'pixel_7_0' 'pixel_7_1' 'pixel_7_5']
# 최고 점수 98.333%