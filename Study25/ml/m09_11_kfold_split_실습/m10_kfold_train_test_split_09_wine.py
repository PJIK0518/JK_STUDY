
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
    random_state=777
)

# print(x.shape, y.shape) (178, 13) (178,)
n_split = 5
KFold = KFold(n_splits=n_split,
              shuffle=True,
              random_state=333)

stfKFold = StratifiedKFold(n_splits=n_split, # label 을 균형있께 뽑아줌 : 통상적으로 분류 데이터에서 효과적
                           shuffle=True,
                           random_state=333)

#2. 모델
model = HistGradientBoostingClassifier()
# model = RandomForestRegressor()

#3. 훈련
scores = cross_val_score(model, x_trn, y_trn, cv=stfKFold)                         # compile, fit이 한 번에 진행되는 형태
print(model)
print('ACC :', scores , '\n평균 ACC :', np.round(np.mean(scores), 4))

y_prd = cross_val_predict(model, x_tst, y_tst, cv=stfKFold)
score = accuracy_score(y_tst, y_prd)

print(score)

# kfold_tts 0.2777777777777778
""" HistGradientBoostingClassifier()
ACC : [0.97222222 0.88888889 0.97222222 1.         0.97142857]
평균 ACC : 0.961 """

# LinearSVC(C=0.3) :  0.9629629629629629
# LogisticRegression() :  0.9259259259259259
# DecisionTreeClassifier() :  0.8703703703703703
# RandomForestClassifier() :  0.9814814814814815