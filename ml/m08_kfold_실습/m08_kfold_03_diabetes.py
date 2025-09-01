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

from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
x, y = load_diabetes(return_X_y=True)

MS = MinMaxScaler()

MS.fit(x)

x_trn = MS.transform(x)
x_tst = MS.transform(x)

from tensorflow.python.keras.layers import Conv2D, Flatten
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

#4. 평가
from sklearn.metrics import mean_squared_error
y_prd = model.predict(x)
loss = mean_squared_error(y, y_prd)

print(model)
print('ACC :', scores , '\n평균 ACC :', np.round(np.mean(scores), 4))

""" HistGradientBoostingRegressor()
ACC : [0.32241346 0.36710717 0.47591973 0.25545567 0.36580256]
평균 ACC : 0.3573 
"""

''' loss
3963.572265625
[DO]
3465.424072265625
[CNN]
2477.224609375
[LSTM]
4012.264404296875
[Conv1D]
2314.77783203125
'''
