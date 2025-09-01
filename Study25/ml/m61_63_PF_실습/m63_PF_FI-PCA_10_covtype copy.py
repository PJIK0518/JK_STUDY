# m10_10.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#       >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#       >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#       >> 데이터의 소실 없이 훈련 가능

import warnings

warnings.filterwarnings('ignore')


import numpy as np
import pandas as pdc

from sklearn.datasets import fetch_covtype
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
DS = fetch_covtype()

x = DS.data
y = DS.target

y = y-1

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import pandas as pd

x = pd.DataFrame(x, columns=DS.feature_names)

PF = PolynomialFeatures(degree=2, include_bias=False)
x_trn = PF.fit_transform(x_trn)
x_tst = PF.transform(x_tst)

poly_feature_names = PF.get_feature_names_out(input_features=x.columns)

x_trn = pd.DataFrame(x_trn, columns=poly_feature_names)
x_tst = pd.DataFrame(x_tst, columns=poly_feature_names)

# 🚩 2️⃣ Feature Importance 평가 (XGBRegressor 사용)
model_fi = XGBRegressor(random_state=42)
model_fi.fit(x_trn, y_trn)

FI = model_fi.feature_importances_
CPT = np.percentile(FI, 25)

# 🚩 3️⃣ 중요도 기준 feature 분리
low_FI_cols = [col for col, imp in zip(x.columns, FI) if imp <= CPT]
high_FI_cols = [col for col, imp in zip(x.columns, FI) if imp > CPT]

x_trn_high = x_trn[high_FI_cols].to_numpy()
x_tst_high = x_tst[high_FI_cols].to_numpy()

x_trn_low = x_trn[low_FI_cols].to_numpy()
x_tst_low = x_tst[low_FI_cols].to_numpy()

print(f"# Feature 분리 결과: High={len(high_FI_cols)} / Low={len(low_FI_cols)}")

# 🚩 4️⃣ 하위 feature PCA 적용
PCA_model = PCA()
PCA_model.fit(x_trn_low)

EVR_cumsum = np.cumsum(PCA_model.explained_variance_ratio_)
n_components = np.argmax(EVR_cumsum >= 0.95) + 1

print(f"# PCA 선택된 n_components = {n_components}")

PCA_model = PCA(n_components=n_components)
x_trn_pca = PCA_model.fit_transform(x_trn_low)
x_tst_pca = PCA_model.transform(x_tst_low)

# 🚩 5️⃣ 최종 feature 합치기
x_trn_final = np.hstack([x_trn_high, x_trn_pca])
x_tst_final = np.hstack([x_tst_high, x_tst_pca])

print(f"# 최종 x_trn shape: {x_trn_final.shape}")
print(f"# 최종 x_tst shape: {x_tst_final.shape}")

# 🚩 6️⃣ LinearRegression으로 학습 및 평가
model_final = LinearRegression()
model_final.fit(x_trn_final, y_trn)
y_pred = model_final.predict(x_tst_final)
score = r2_score(y_tst, y_pred)

print(f"# 최종 LinearRegression R2 Score: {score:.4f}")

# Feature 분리 결과: High=19 / Low=35
# PCA 선택된 n_components = 18
# 최종 x_trn shape: (522910, 37)
# 최종 x_tst shape: (58102, 37)
# 최종 LinearRegression R2 Score: 0.3160

# PF R2 : 0.4816085978219853
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9722901104953358
# ACC : 0.9722901104953358
# F1S : 0.9432327280233942

# ORIG_SCR : 0.8671302192695605
# Quantile : 0.25
# DROP_SCR : 0.8840863715491213
# ORIG_SCR : 0.8671302192695605
# DROP_COL : ['Slope', 'Soil_Type_6', 'Soil_Type_7',
#             'Soil_Type_8', 'Soil_Type_14', 'Soil_Type_15',
#             'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_20',
#             'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26',
#             'Soil_Type_27', 'Soil_Type_35']

""" R2 : 0.9174727203882826
Threshold = 0.002 / n = 54 / R2 = 91.747%
Threshold = 0.003 / n = 53 / R2 = 91.747%
Threshold = 0.004 / n = 52 / R2 = 91.780%
Threshold = 0.004 / n = 51 / R2 = 91.766%
Threshold = 0.004 / n = 50 / R2 = 91.871%
Threshold = 0.006 / n = 49 / R2 = 91.813%
Threshold = 0.006 / n = 48 / R2 = 91.937%
Threshold = 0.006 / n = 47 / R2 = 91.897%
Threshold = 0.006 / n = 46 / R2 = 91.777%
Threshold = 0.006 / n = 45 / R2 = 91.983%
Threshold = 0.007 / n = 44 / R2 = 91.835%
Threshold = 0.007 / n = 43 / R2 = 91.895%
Threshold = 0.008 / n = 42 / R2 = 91.959%
Threshold = 0.008 / n = 41 / R2 = 91.677%
Threshold = 0.008 / n = 40 / R2 = 90.861%
Threshold = 0.009 / n = 39 / R2 = 90.851%
Threshold = 0.010 / n = 38 / R2 = 89.933%
Threshold = 0.010 / n = 37 / R2 = 89.913%
Threshold = 0.011 / n = 36 / R2 = 89.950%
Threshold = 0.012 / n = 35 / R2 = 90.135%
Threshold = 0.012 / n = 34 / R2 = 90.181%
Threshold = 0.012 / n = 33 / R2 = 87.923%
Threshold = 0.012 / n = 32 / R2 = 87.851%
Threshold = 0.012 / n = 31 / R2 = 80.130%
Threshold = 0.012 / n = 30 / R2 = 80.207%
Threshold = 0.013 / n = 29 / R2 = 80.133%
Threshold = 0.013 / n = 28 / R2 = 72.789%
Threshold = 0.014 / n = 27 / R2 = 72.629%
Threshold = 0.015 / n = 26 / R2 = 72.553%
Threshold = 0.015 / n = 25 / R2 = 72.600%
Threshold = 0.016 / n = 24 / R2 = 72.603%
Threshold = 0.016 / n = 23 / R2 = 72.564%
Threshold = 0.018 / n = 22 / R2 = 72.450%
Threshold = 0.018 / n = 21 / R2 = 72.438%
Threshold = 0.020 / n = 20 / R2 = 72.247%
Threshold = 0.021 / n = 19 / R2 = 72.242%
Threshold = 0.021 / n = 18 / R2 = 71.836%
Threshold = 0.022 / n = 17 / R2 = 71.820%
Threshold = 0.022 / n = 16 / R2 = 71.731%
Threshold = 0.023 / n = 15 / R2 = 71.734%
Threshold = 0.024 / n = 14 / R2 = 71.650%
Threshold = 0.026 / n = 13 / R2 = 71.622%
Threshold = 0.027 / n = 12 / R2 = 71.593%
Threshold = 0.028 / n = 11 / R2 = 71.509%
Threshold = 0.031 / n = 10 / R2 = 70.803%
Threshold = 0.031 / n =  9 / R2 = 70.605%
Threshold = 0.034 / n =  8 / R2 = 70.189%
Threshold = 0.034 / n =  7 / R2 = 70.003%
Threshold = 0.037 / n =  6 / R2 = 69.385%
Threshold = 0.041 / n =  5 / R2 = 69.101%
Threshold = 0.048 / n =  4 / R2 = 68.292%
Threshold = 0.052 / n =  3 / R2 = 68.283%
Threshold = 0.061 / n =  2 / R2 = 67.721%
Threshold = 0.064 / n =  1 / R2 = 48.584% """

