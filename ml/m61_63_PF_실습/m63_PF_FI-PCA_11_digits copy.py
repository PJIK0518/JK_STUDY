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

# Feature 분리 결과: High=52 / Low=12
# PCA 선택된 n_components = 3
# 최종 x_trn shape: (1617, 55)
# 최종 x_tst shape: (180, 55)
# 최종 LinearRegression R2 Score: 0.6072

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