# m08_00.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#       >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#       >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#       >> 데이터의 소실 없이 훈련 가능

import warnings

warnings.filterwarnings('ignore')


import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

n_split = 5
KFold = KFold(n_splits=n_split,
              shuffle=True,
              random_state=333)

stfKFold = StratifiedKFold(n_splits=n_split, # label 을 균형있께 뽑아줌 : 통상적으로 분류 데이터에서 효과적
                           shuffle=True,
                           random_state=333)

#2. 모델
model = HistGradientBoostingRegressor()
# model = RandomForestRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=KFold)                         # compile, fit이 한 번에 진행되는 형태

scores = model.evaluate(x, y)

print(model)
print('ACC :', scores , '\n평균 ACC :', np.round(np.mean(scores), 4))

""" HistGradientBoostingRegressor()
ACC : [0.82830453 0.8374145  0.82551518 0.84078563 0.84934257]
평균 ACC : 0.8363 """

''' loss
0.47339358925819397
[DO]
0.3632981777191162
[CNN]
0.5098786354064941
0.4542386829853058
0.3375522494316101
[LSTM]
0.44303375482559204
[Conv1D]
0.4965818524360657
'''