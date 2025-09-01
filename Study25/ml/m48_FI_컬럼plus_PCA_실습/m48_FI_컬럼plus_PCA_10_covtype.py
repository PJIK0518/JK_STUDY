# m10_10.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#       >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#       >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#       >> 데이터의 소실 없이 훈련 가능

import warnings

warnings.filterwarnings('ignore')


import numpy as np
import pandas as pdc

from sklearn.datasets import fetch_covtype
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
DS = fetch_covtype()

x = DS.data
y = DS.target

y = y-1

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
from xgboost import XGBRegressor, XGBClassifier
import pandas as pd

Q = 25
#2 모델구성
model = XGBClassifier(random_state=24)

model.fit(x_trn, y_trn)

print(model.feature_importances_)

print('ORIG_SCR :', model.score(x_tst, y_tst))
                                                                     
CPT = np.percentile(model.feature_importances_, Q)

COL_name = []

for i, FI in enumerate(model.feature_importances_):
    if FI <= CPT:
        COL_name.append(DS.feature_names[i])
    else:
        continue

x_df = pd.DataFrame(x, columns=DS.feature_names)
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

model.fit(x_trn, y_trn)

score = model.score(x_trn, y_trn)
print('PLUS_SCR :', score)
print('DROP_COL :', COL_name)

# ORIG_SCR : 0.8671302192695605
# Quantile : 0.25
# DROP_SCR : 0.8840863715491213
# ORIG_SCR : 0.8671302192695605
# DROP_COL : ['Slope', 'Soil_Type_6', 'Soil_Type_7',
#             'Soil_Type_8', 'Soil_Type_14', 'Soil_Type_15',
#             'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_20',
#             'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26',
#             'Soil_Type_27', 'Soil_Type_35']