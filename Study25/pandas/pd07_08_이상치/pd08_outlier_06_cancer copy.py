# KNN in Regressor
# 가장 가까운 값들의 y  값에 대한 평균
# 그닥 많이 쓰는 거는 아님
# Classifier와 다르게 k 값이 홀수일 필요가 없음

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import numpy as np
import random

import matplotlib.pyplot as plt

seed = 123
random.seed(seed)
np.random.seed(seed)
    
#1 데이터
DS =load_breast_cancer()
x = DS.data
y = DS.target

print(x.shape, y.shape)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    random_state=seed,
    stratify=y)

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
n = 3
r = 10
fig, axs = plt.subplots(n, r, figsize=(15, 9))
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
col = range(x.shape[1])

for i in col:
    x_trn_col = x_trn[:, i].reshape(-1, 1)
    x_tst_col = x_tst[:, i].reshape(-1, 1)
    
    RSC.fit(x_trn_col)
    x_trn[:, i] = RSC.transform(x_trn_col).reshape(-1)
    x_tst[:, i] = RSC.transform(x_tst_col).reshape(-1)




scaler = StandardScaler()
x_trn = scaler.fit_transform(x_trn)
x_tst = scaler.transform(x_tst)

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

print('# Outlier 처리')
print('R2 :',score1)
# Outlier 처리
# R2 : -30.71868229946188

# PF R2 : -30.718682299459076

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9707602339181286
# ACC : 0.9707602339181286
# F1S : 0.9769585253456221