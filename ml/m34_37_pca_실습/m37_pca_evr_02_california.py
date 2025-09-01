import numpy as np
import random
import warnings

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

RS = 35
np.random.seed(RS)
random.seed(RS)

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.8,
    random_state=RS,  
)

from sklearn.decomposition import PCA
import time

x_trn = x_trn.reshape(x_trn.shape[0],-1)

pca = PCA(n_components=x_trn.shape[1])

x_trn = pca.fit_transform(x_trn)
x_tst = pca.fit_transform(x_tst)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print(evr_cumsum)
a = int(x_trn.shape[1]) - len(np.where(evr_cumsum >= 1.)[0]) + 1
b = int(x_trn.shape[1]) - len(np.where(evr_cumsum >= 0.999)[0]) + 1
c = int(x_trn.shape[1]) - len(np.where(evr_cumsum >= 0.99)[0]) + 1
d = int(x_trn.shape[1]) - len(np.where(evr_cumsum >= 0.95)[0]) + 1
print(x_trn.shape)
print(a,b,c,d)
num = [a,b,c,d]
acc = []
exit()

for p in num :
    pca = PCA(n_components=p)
    pca.fit(x_trn)
    x_trn_P = pca.transform(x_trn)
    x_tst_P = pca.transform(x_tst)
    
    S = time.time()

    #2. 모델
    from catboost import CatBoostRegressor, CatBoostClassifier

    model = CatBoostRegressor(verbose=0)
    
    #3. 컴파일 훈련
    model.fit(x_trn_P, y_trn)
    
    score = model.score(x_tst_P, y_tst)
    
    acc.append((p, score))

for p, ACC in acc:
    print(f"n_components={p:>3} | r2={float(ACC):.4f}")
    
# n_components=  1 | r2=-0.0133
# n_components=  8 | r2=-0.0907
# n_components=  8 | r2=-0.0907
# n_components=  8 | r2=-0.0907

# CatBoostRegressor R2 : 0.8492
