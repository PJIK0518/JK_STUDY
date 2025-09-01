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

def outlier(data):
    out = []
    up = []
    low = []
    for i in range(data.shape[1]):
        col = data[:, i]
        Q1, Q3 = np.percentile(col, [25, 75])
        
        IQR = Q3 - Q1
        print('IQR :', IQR)
        
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        
        out_i = np.where((col > upper_bound) | (col < lower_bound))[0]
        out.append(out_i)
        up.append(upper_bound)
        low.append(lower_bound)
    return out, up, low

OUT, UP, LOW = outlier(x)

print(OUT) # [array([ 0, 12]), array([6])]
print(UP)  # [19.0, 1200.0]
print(LOW) # [-5.0, -400.0]

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, x.shape[1], figsize=(15, 9))

for i in range(x.shape[1]):
    axs[i].boxplot(x[:,i])
    axs[i].axhline(UP[i], color = 'red', label = 'upper_bound')
    axs[i].axhline(LOW[i], color = 'red', label = 'lower_bound')
    axs[i].set_title(f"Column {i}")
    
# plt.tight_layout()
# plt.show()
# exit()

from sklearn.preprocessing import RobustScaler
RSC = RobustScaler()
col = [1,2,3,4,8,9,10]

for i in col:
    x_trn_col = x_trn[:, i].reshape(-1, 1)
    x_tst_col = x_tst[:, i].reshape(-1, 1)
    
    RSC.fit(x_trn_col)
    x_trn[:, i] = RSC.transform(x_trn_col).reshape(-1)
    x_tst[:, i] = RSC.transform(x_tst_col).reshape(-1)


print('# Outlier 처리')

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

from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.preprocessing import PolynomialFeatures
PF = PolynomialFeatures(degree=2, include_bias=False)
x_trn = PF.fit_transform(x_trn)
x_tst = PF.transform(x_tst)

#2 모델구성
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('R2 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

score1 = r2_score(y_tst, y_prd)

print('R2 :',score1)
# Outlier 처리
# R2 : 0.8993897305817133
# R2 : 0.8993897306640561
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