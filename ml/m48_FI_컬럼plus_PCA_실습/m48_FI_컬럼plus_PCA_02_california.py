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
# print(x.shape, y.shape) (20640, 8) (20640,)

from xgboost import XGBRegressor, XGBClassifier
import pandas as pd

Q = 25
#2 모델구성
model = XGBRegressor(random_state=24)

model.fit(x_trn, y_trn)

print(model.feature_importances_)

# [0.46729922 0.06602691 0.04730553 0.02748645 0.02454475 0.1568825
#  0.10187978 0.10857485]

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
x2 = x_df[['AveBedrms', 'Population']]
# print(x2)

x1_trn, x1_tst, x2_trn, x2_tst, y_trn, y_tst \
    = train_test_split(x1, x2, y,
                       train_size=0.7,
                       random_state=42)

# print('', x1_trn.shape, x1_tst.shape,'\n',
#       x2_trn.shape, x2_tst.shape,'\n',
#       y_trn.shape, y_tst.shape)
#  (14447, 6) (6193, 6) 
#  (14447, 2) (6193, 2) 
#  (16512,) (4128,)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x2_trn = pca.fit_transform(x2_trn)
x2_tst = pca.transform(x2_tst)

print(x2_trn.shape, x2_tst.shape)
# (14447, 1) (6193, 1)
x_trn = np.concatenate([x1_trn, x2_trn], axis=1)
x_tst = np.concatenate([x1_tst, x2_tst], axis=1)

model.fit(x_trn, y_trn)

score = model.score(x_trn, y_trn)
print('Quantile :', Q/100)
print('DROP_SCR :', score)
print('DROP_COL :', COL_name)

# [0.46729922 0.06602691 0.04730553 0.02748645 0.02454475 0.1568825
#  0.10187978 0.10857485]
# ORIG_SCR : 0.8362797172573576
# (14447, 1) (6193, 1)
# Quantile : 0.25
# DROP_SCR : 0.9447257725490363
# DROP_COL : ['AveBedrms', 'Population']