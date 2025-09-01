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
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier

#1. 데이터
DS = load_digits()

x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
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
fig, axs = plt.subplots(8,8, figsize=(15, 9))
axs = axs.flatten()

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
col = [1, 5, 7,9, 14, 15,
        17, 22, 23,
        25, 30, 33,
        40, 41, 47,
        48, 49, 55, 56, 57, 59, 62, 63]

for i in col:
    x_trn_col = x_trn[:, i].reshape(-1, 1)
    x_tst_col = x_tst[:, i].reshape(-1, 1)
    
    RSC.fit(x_trn_col)
    x_trn[:, i] = RSC.transform(x_trn_col).reshape(-1)
    x_tst[:, i] = RSC.transform(x_tst_col).reshape(-1)


print('# Outlier 처리')

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
# R2 : -4.402188759880493

# PF R2 : -2.0539186331728154
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9777777777777777
# ACC : 0.9777777777777777
# F1S : 0.9771982808167019


# 최적 컬럼 : 23 개 
#  ['pixel_0_5' 'pixel_1_4' 'pixel_1_5' 'pixel_2_3' 'pixel_2_5' 'pixel_3_2'
#  'pixel_3_4' 'pixel_4_1' 'pixel_4_4' 'pixel_4_5' 'pixel_4_6' 'pixel_5_2'
#  'pixel_5_3' 'pixel_5_5' 'pixel_6_0' 'pixel_6_3' 'pixel_6_5' 'pixel_6_6'
#  'pixel_7_2' 'pixel_7_3' 'pixel_7_4' 'pixel_7_6' 'pixel_7_7']
# 삭제 컬럼 : 41 개 
#  ['pixel_0_0' 'pixel_0_1' 'pixel_0_2' 'pixel_0_3' 'pixel_0_4' 'pixel_0_6'
#  'pixel_0_7' 'pixel_1_0' 'pixel_1_1' 'pixel_1_2' 'pixel_1_3' 'pixel_1_6'
#  'pixel_1_7' 'pixel_2_0' 'pixel_2_1' 'pixel_2_2' 'pixel_2_4' 'pixel_2_6'
#  'pixel_2_7' 'pixel_3_0' 'pixel_3_1' 'pixel_3_3' 'pixel_3_5' 'pixel_3_6'
#  'pixel_3_7' 'pixel_4_0' 'pixel_4_2' 'pixel_4_3' 'pixel_4_7' 'pixel_5_0'
#  'pixel_5_1' 'pixel_5_4' 'pixel_5_6' 'pixel_5_7' 'pixel_6_1' 'pixel_6_2'
#  'pixel_6_4' 'pixel_6_7' 'pixel_7_0' 'pixel_7_1' 'pixel_7_5']
# 최고 점수 98.333%