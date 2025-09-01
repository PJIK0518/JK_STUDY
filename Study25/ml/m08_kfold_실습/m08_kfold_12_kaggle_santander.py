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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


#1. 데이터
path = './_data/kaggle/santander/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

x = trn_csv.drop(['target'], axis=1)
y = trn_csv['target']

#####################################
## Scaler
def Scaler(SC, a, b):
    SC.fit(a)
    return SC.transform(a), SC.transform(b)

x, tst_csv = Scaler(RobustScaler(), x, tst_csv)

def Scaler(SC, a, b):
    SC.fit(a)
    return SC.transform(a), SC.transform(b)

x, tst_csv = Scaler(StandardScaler(), x, tst_csv)

#####################################
## 증폭 : class_weight
classes = np.unique(y)
CW = compute_class_weight(class_weight='balanced', classes=classes, y=y)

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
scores = cross_val_score(model, x, y, cv=stfKFold)                         # compile, fit이 한 번에 진행되는 형태
print(model)
print('ACC :', scores , '\n평균 ACC :', np.round(np.mean(scores), 4))

""" HistGradientBoostingClassifier()
ACC : [0.9074   0.90745  0.908325 0.908225 0.907925]
평균 ACC : 0.9079 """

# LinearSVC(C=0.3) :  0.9109
# LogisticRegression() :  0.91342
# DecisionTreeClassifier() :  0.83109
# RandomForestClassifier() :  0.89863