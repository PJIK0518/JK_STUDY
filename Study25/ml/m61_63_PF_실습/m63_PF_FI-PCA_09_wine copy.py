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

# Feature 분리 결과: High=6 / Low=7
# PCA 선택된 n_components = 1
# 최종 x_trn shape: (160, 7)
# 최종 x_tst shape: (18, 7)
# 최종 LinearRegression R2 Score: 0.6626

# PF R2 : 0.8993897306640561
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