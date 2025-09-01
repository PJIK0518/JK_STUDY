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
from sklearn.decomposition import PCA
import time
from xgboost import XGBRegressor, XGBClassifier
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier
import pandas as pd

Q = 25
#2 모델구성
model = XGBClassifier(random_state=24)

#####################################
## Scaler
def Scaler(SC, a, b):
    SC.fit(a)
    a_scaled = SC.transform(a)
    b_scaled = SC.transform(b)
    return a_scaled, b_scaled

x_trn, x_tst = Scaler(RobustScaler(), x_trn, x_tst)
x_trn, x_tst = Scaler(StandardScaler(), x_trn, x_tst)

model.fit(x_trn, y_trn)

print(model.feature_importances_)

print('ORIG_SCR :', model.score(x_tst, y_tst))
                                                                     
CPT = np.percentile(model.feature_importances_, Q)

COL_name = []

for i, FI in enumerate(model.feature_importances_):
    if FI <= CPT:
        COL_name.append(trn_csv.columns[i])
    else:
        continue

x_df = pd.DataFrame(x, columns=trn_csv.columns)
x1 = x_df.drop(columns=COL_name)
x2 = x_df[COL_name]

x1_trn, x1_tst, x2_trn, x2_tst, y_trn, y_tst \
    = train_test_split(x1, x2, y,
                       train_size=0.7,
                       random_state=42)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x2_trn = pca.fit_transform(x2_trn)
x2_tst = pca.transform(x2_tst)

print(x2_trn.shape, x2_tst.shape)
x_trn = np.concatenate([x1_trn, x2_trn], axis=1)
x_tst = np.concatenate([x1_tst, x2_tst], axis=1)

#####################################
## Scaler
def Scaler(SC, a, b):
    SC.fit(a)
    a_scaled = SC.transform(a)
    b_scaled = SC.transform(b)
    return a_scaled, b_scaled

x_trn, x_tst = Scaler(RobustScaler(), x_trn, x_tst)
x_trn, x_tst = Scaler(StandardScaler(), x_trn, x_tst)

model.fit(x_trn, y_trn)

score = model.score(x_trn, y_trn)
print('PLUS_SCR :', score)
print('DROP_COL :', COL_name)

# ORIG_SCR : 0.9161
# Quantile : 0.25
# DROP_SCR : 0.9555357142857143
# PLUS_SCR : 0.9556071428571429