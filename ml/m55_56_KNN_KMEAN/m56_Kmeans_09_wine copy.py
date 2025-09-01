# m10_09.copy

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
    random_state=777,
    stratify=y
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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
#2 모델구성
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, f1_score

nc = 3
model1 = KMeans(n_clusters=nc,      
               init='k-means++',   
               n_init=50,

               )
model2 = KMeans(n_clusters=nc,     
               init='k-means++',
               n_init=50,         
               random_state=34,
               )
model3 = KMeans(n_clusters=nc,       
               init='k-means++',   
               n_init=50,         
               random_state=56,
               )
model4 = KMeans(n_clusters=nc,       
               init='k-means++',   
               n_init=50,         
               random_state=78,
               )
model5 = KMeans(n_clusters=nc,       
               init='k-means++',   
               n_init=50,         
               random_state=90,
               )

y_trn_prd1 = model1.fit_predict(x_trn)
y_tst_prd1 = model1.predict(x_tst)

y_trn_prd2 = model2.fit_predict(x_trn)
y_tst_prd2 = model2.predict(x_tst)

y_trn_prd3 = model3.fit_predict(x_trn)
y_tst_prd3 = model3.predict(x_tst)

y_trn_prd4 = model4.fit_predict(x_trn)
y_tst_prd4 = model4.predict(x_tst)

y_trn_prd5 = model5.fit_predict(x_trn)
y_tst_prd5 = model5.predict(x_tst)

y_votes = np.vstack([y_tst_prd1, y_tst_prd2, y_tst_prd3, y_tst_prd4, y_tst_prd5])

from scipy.stats import mode
y_vote_result, _ = mode(y_votes, axis=0, keepdims=False)

score1 = accuracy_score(y_tst, y_vote_result)
score2 = f1_score(y_tst, y_vote_result, average='macro')

print('✅ Manual Hard Voting for KMeans ensemble')
print('ACC :', score1)
print('F1S :', score2)

# ACC : 0.16666666666666666
# F1S : 0.16666666666666666

#  ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.6666666666666666
# ACC : 0.6666666666666666
# F1S : 0.6031746031746031   
    
# 최적 컬럼 : 3 개 
#  ['color_intensity' 'od280/od315_of_diluted_wines' 'proline']
# 삭제 컬럼 : 10 개 
#  ['alcohol' 'malic_acid' 'ash' 'alcalinity_of_ash' 'magnesium'
#  'total_phenols' 'flavanoids' 'nonflavanoid_phenols' 'proanthocyanins'
#  'hue']
# 최고 점수 100.000%