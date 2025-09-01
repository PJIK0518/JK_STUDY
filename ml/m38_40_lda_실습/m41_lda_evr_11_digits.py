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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

#1. 데이터
DS = load_digits()

x = DS.data
y = DS.target

y_org = y.copy()

y = np.rint(y).astype(int) # 기존 부동소수점 형태에서 > 정수형

# print(np.unique(y, return_counts=True))

x_trn, x_tst, y_trn, y_tst, y_trn_org, y_tst_org= train_test_split(
    x, y, y_org,
    train_size=0.7,
    random_state=42
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

warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
import time

############################## PCA ##############################
# from sklearn.decomposition import PCA
# pca = PCA(n_components=9)

# pca.fit_transform(x_trn, y_trn)
# x_trn = pca.transform(x_trn)
# x_tst = pca.transform(x_tst)
# pca_evr = np.cumsum(pca.explained_variance_ratio_)
""" print(pca_evr)
[0.14870031 0.28501313 0.40508943 0.48968487 0.54607463 0.5953106
 0.63793755 0.6736126  0.70700127 0.73797757 0.76184203 0.78502564
 0.80348655 0.82109603 0.8360106  0.85045051 0.86379308 0.87582449
 0.88600703 0.89536366 0.90422936 0.91217338 0.91955718 0.92677681
 0.93339348 0.93943522 0.94521632 0.95029174 0.95509394 0.95934885
 0.96319955 0.966691   0.9699912  0.9732216  0.97620124 0.97912899
 0.9816758  0.98399743 0.9862584  0.98840512 0.99025768 0.99180309
 0.99327799 0.99464551 0.99577605 0.99684093 0.9977711  0.99861656
 0.99913294 0.99951526 0.99974448 0.99982251 0.99988282 0.99992979
 0.99996989 0.99998648 0.99999403 0.99999779 0.99999882 0.99999956
 1.         1.         1.         1.        ] """

############################## LDA ##############################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=9)

lda.fit_transform(x_trn, y_trn)
x_trn = lda.transform(x_trn)
x_tst = lda.transform(x_tst)


#2. model
model = RandomForestClassifier(random_state=2)
# model = RandomForestRegressor(random_state=32131)

#3. 훈련
model.fit(x_trn, y_trn_org)

#4 평가
result = model.score(x_tst, y_tst_org)

y_prd = model.predict(x_trn)

print(np.unique(y_trn_org,return_counts=True))
print(np.unique(y_prd,return_counts=True))

print('점수 :', result)
# ORG 점수 : 0.8324707904045475
# n_component = 20
# PCA 점수 : 0.8184515979548649
# LDA 점수 : 0.8767181142742396


    
# n_components=  1 | acc=0.0889
# n_components= 16 | acc=0.3111
# n_components= 25 | acc=0.2833
# n_components= 38 | acc=0.3111
    
# CatBoostClassifier | train pred shape: (1617, 1), test pred shape: (180, 1)
# XGBClassifier | train pred shape: (1617,), test pred shape: (180,)
# RandomForestClassifier | train pred shape: (1617,), test pred shape: (180,)
# LGBMClassifier | train pred shape: (1617,), test pred shape: (180,)
# Stacking score : 0.9722222222222222
# hard: 0.9777777777777777
# soft: 0.9833333333333333