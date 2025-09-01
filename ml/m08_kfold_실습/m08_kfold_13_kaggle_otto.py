# m08_00.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#           >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#           >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#           >> 데이터의 소실 없이 훈련 가능

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
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

#1. 데이터
path = 'C:/Study25/_data/kaggle/otto/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = trn_csv.drop(['target'], axis=1)
y = trn_csv['target']

#####################################
## Scaler
def Scaler(SC, x, tst_csv):
    SC.fit(x)
    return SC.transform(x), SC.transform(tst_csv)

x,  tst_csv = Scaler(MaxAbsScaler(), x, tst_csv)

#####################################
## 증폭 : class_weight
classes = np.unique(y)
CW = compute_class_weight(class_weight='balanced', classes=classes, y=y)

#####################################
## Label_y
LE = LabelEncoder()
LE.fit(y)
y_trn = LE.transform(y)

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
ACC : [0.8125404  0.81456044 0.80963154 0.8119596  0.81147475]
평균 ACC : 0.812 """

# LinearSVC(C=0.3) :  0.73499466692524
# LogisticRegression() :  0.7371925401596691
# DecisionTreeClassifier() :  0.7009599534568021
# RandomForestClassifier() :  0.7991208507062284