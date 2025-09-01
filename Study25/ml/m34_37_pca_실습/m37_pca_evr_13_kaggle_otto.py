# m10_13.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#           >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#           >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#           >> 데이터의 소실 없이 훈련 가능
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


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
path = './Study25/_data/kaggle/otto/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = trn_csv.drop(['target'], axis=1)
y = trn_csv['target']

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

from sklearn.decomposition import PCA
import time

pca = PCA(n_components=x_trn.shape[1])

#####################################
## Scaler
def Scaler(SC, a, b):
    SC.fit(a)
    a_scaled = pd.DataFrame(SC.transform(a), columns=a.columns, index=a.index)
    b_scaled = pd.DataFrame(SC.transform(b), columns=a.columns, index=b.index)
    return a_scaled, b_scaled

x_trn, tst_csv = Scaler(MaxAbsScaler(), x_trn, tst_csv)

#####################################
## 증폭 : class_weight
classes = np.unique(y)
CW = compute_class_weight(class_weight='balanced', classes=classes, y=y)

#####################################
## Label_y
LE = LabelEncoder()
LE.fit(y_trn)
y_trn = LE.transform(y_trn)
y_tst = LE.transform(y_tst)

x_trn = pca.fit_transform(x_trn)
x_tst = pca.fit_transform(x_tst)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

a = len(np.where(evr_cumsum >= 1.)[0])
b = len(np.where(evr_cumsum >= 0.999)[0])
c = len(np.where(evr_cumsum >= 0.99)[0])
d = len(np.where(evr_cumsum >= 0.95)[0])

num = [a,b,c,d]
acc = []

for p in num :
    pca = PCA(n_components=p)
    pca.fit(x_trn)
    x_trn_P = pca.transform(x_trn)
    x_tst_P = pca.transform(x_tst)
    
    S = time.time()

    #2. 모델
    from catboost import CatBoostRegressor, CatBoostClassifier

    model = CatBoostClassifier(verbose=0)
    
    #3. 컴파일 훈련
    model.fit(x_trn_P, y_trn)
    
    score = model.score(x_tst_P, y_tst)
    
    acc.append((p, score))

for p, ACC in acc:
    print(f"n_components={p:>3} | acc={float(ACC):.4f}")
    
# CatBoostClassifier | train pred shape: (55690, 1), test pred shape: (6188, 1)
# XGBClassifier | train pred shape: (55690,), test pred shape: (6188,)
# RandomForestClassifier | train pred shape: (55690,), test pred shape: (6188,)
# LGBMClassifier | train pred shape: (55690,), test pred shape: (6188,)
# Stacking score : 0.6045572074983839

# {'target': 0.3361253998373447,
#  'params': {'colsample_bylevel': 0.6150592313147155,
#             'colsample_bytree': 1.0,
#             'gamma': 5.0, 'learning_rate': 0.1,
#             'max_bin': 343.4072919726811,
#             'max_depth': 3.0,
#             'min_child_samples': 43.192163305392114,
#             'min_child_weight': 1.0,
#             'n_estimator': 456.0235062465993,
#             'num_leaves': 33.260173446440554,
#             'reg_alpha': 8.32230482255393,
#             'reg_lambda': 25.320411088502198,
#             'subsample': 1.0}}