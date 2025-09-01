# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#       >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#       >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#       >> 데이터의 소실 없이 훈련 가능

import warnings

warnings.filterwarnings('ignore')


import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#1. 데이터
DS = load_iris()

x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)

SDS = StandardScaler()
SDS.fit(x_trn)
x_trn = SDS.transform(x_trn)
x_tst = SDS.transform(x_tst)

NS = 3
stfKF = StratifiedKFold(n_splits=NS, shuffle=True, random_state=777)

model = MLPClassifier()

score = cross_val_score(model, x_trn, y_trn, cv=stfKF)

print('ACC :', score)
print('MACC:', np.round(np.mean(score), 5))

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
y_prd = cross_val_predict(model, x_tst, y_tst, cv=stfKF)
print(y_tst)
print(y_prd)

ACC = accuracy_score(y_tst, y_prd)

print(ACC)