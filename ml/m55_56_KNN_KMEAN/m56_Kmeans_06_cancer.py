# KNN : 이미 알고 있는 label 값에 대해 예측해야하는 값을 거리 기준으로 예측
# KMEAN : label을 모르고 있다면 >> Clustering 방식 - 군집방식
        # 임의의 데이터에 label 값 설정
        # 기존 데이터와의 거리가 더 가까운 임의의 데이텅 




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

scaler = StandardScaler()
x_trn = scaler.fit_transform(x_trn)
x_tst = scaler.transform(x_tst)

#2 모델구성
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, f1_score
model1 = KMeans(n_clusters=2,        # label의 개수로   
               init='k-means++',    # 임의의 값을 최대거리 기존 데이터로 이동
               n_init=10,           # 임의의 값의 이동 횟수
               random_state=123,
               )
model2 = KMeans(n_clusters=2,        # label의 개수로   
               init='k-means++',    # 임의의 값을 최대거리 기존 데이터로 이동
               n_init=10,           # 임의의 값의 이동 횟수
               random_state=456,
               )
model3 = KMeans(n_clusters=2,        # label의 개수로   
               init='k-means++',    # 임의의 값을 최대거리 기존 데이터로 이동
               n_init=10,           # 임의의 값의 이동 횟수
               random_state=789,
               )

## 성능 개판 >> 기존 label이 있는 데이터의 경우에 기존 값이랑 일치하지 않으면...!
# [0 0 0 0 1 1 1 0 0 0]
# [1 1 1 1 0 0 0 1 1 1]
### Label 이 없을때 clustering 방식으로 할 수는 있지만...
#1. 성능이 구린데 구지??
#2. 요즘 그렇게 제공되는 데이터가 없는 편 >> 이미 있는데 구지??


y_trn_prd1 = model1.fit_predict(x_trn)
y_tst_prd1 = model1.predict(x_tst)

y_trn_prd2 = model2.fit_predict(x_trn)
y_tst_prd2 = model2.predict(x_tst)

y_trn_prd3 = model3.fit_predict(x_trn)
y_tst_prd3 = model3.predict(x_tst)

y_votes = np.vstack([y_tst_prd1, y_tst_prd2, y_tst_prd3])

from scipy.stats import mode
y_vote_result, _ = mode(y_votes, axis=0, keepdims=False)

score1 = accuracy_score(y_tst, y_vote_result)
score2 = f1_score(y_tst, y_vote_result)

print('✅ Manual Hard Voting for KMeans ensemble')
print('ACC :', score1)
print('F1S :', score2)

# ACC : 0.9239766081871345
# F1S : 0.9406392694063926
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9707602339181286
# ACC : 0.9707602339181286
# F1S : 0.9769585253456221