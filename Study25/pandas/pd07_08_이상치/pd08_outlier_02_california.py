import numpy as np
import random
import warnings

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')

RS = 35
np.random.seed(RS)
random.seed(RS)

#1. 데이터
DS = fetch_california_housing()
x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.8,
    random_state=RS,  
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
    axs[i].legend()
    axs[i].set_title(f"Column {i}")
    
# plt.tight_layout()
# plt.show()

from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler
SCL = MaxAbsScaler()

col = [0, 2, 3, 4, 5]

for i in col:
    x_trn_col = x_trn[:, i].reshape(-1, 1)
    x_tst_col = x_tst[:, i].reshape(-1, 1)
    
    SCL.fit(x_trn_col)
    x_trn[:, i] = SCL.transform(x_trn_col).reshape(-1)
    x_tst[:, i] = SCL.transform(x_tst_col).reshape(-1)

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
print('# Outlier 처리')
print('# R2 :',score1)
# Outlier 처리
# R2 : 0.6656905139551408

# PF
# 02 R2 : 0.665690514019535

# RFR
# R2 : 0.8114202090374049
# R2 : 0.8114202090374049

## x, y 변환
# R2 : 0.7395296108401057
# R2 : 0.7314014418807302

## x만 변환
# R2 : 0.7361044731472235
# R2 : 0.7361044731472235

## y만 변환
# R2 : 0.8280160231711524
# R2 : 0.8057273043107682