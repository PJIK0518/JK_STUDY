# LDA : 
# PCA처럼 차원을 축소하는 방식 중 하나  
# y 값도 만져주는 녀석 >> 지도 학습 
#   How? 새로운 축을 x 데이터의 label끼리 나눌 수 있게 생성
# 기본적으뢰 분류 모델에 대해서 가능, But 회귀형태의 데이터를 구간별로 범주화 시켜서 적용 가능

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

# discriminant : 판별 /// 선형판별분석

#1. 데이터

DS = load_diabetes()
x = DS.data
y = DS.target

# 범주화!
y_org = y.copy()

y = np.rint(y).astype(int) # 기존 부동소수점 형태에서 > 정수형

# print(np.unique(y, return_counts=True))

x_trn, x_tst, y_trn, y_tst, y_trn_org, y_tst_org= train_test_split(
    x, y, y_org,
    train_size=0.7,
    random_state=42
)

Scaler = StandardScaler()
x_trn = Scaler.fit_transform(x_trn)
x_tst = Scaler.transform(x_tst)

############################## ORG ##############################
# #2. model
# model = RandomForestClassifier(random_state=2)

# #3. 훈련
# model.fit(x_trn, y_trn)

# #4 평가
# y_tst = model.predict(x_tst)

# result = model.score(x_tst, y_tst)

# print('ORG 점수 :', result)

# # ORG 점수 : 1.0

############################## PCA ##############################
# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)

# pca.fit_transform(x_trn, y_trn)
# x_trn = pca.transform(x_trn)
# x_tst = pca.transform(x_tst)
# pca_evr = np.cumsum(pca.explained_variance_ratio_)
# """ print(pca_evr)
# [0.39568206 0.54279937 0.66835444 0.76673712 0.83458907 0.89427628
#  0.94779274 0.991339   0.99905309 1.        ] """

############################## LDA ##############################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# lda = LDA(n_components=10)

# lda.fit_transform(x_trn, y_trn)
# x_trn = lda.transform(x_trn)
# x_tst = lda.transform(x_tst)
# lda_evr = np.cumsum(lda.explained_variance_ratio_)
# """ print(lda_evr)
# [0.2942619  0.43324544 0.53994885 0.63148948 0.71316311 0.78661055
#  0.85201706 0.90686947 0.96008579 1.        ] """

#2. model
# model = RandomForestClassifier(random_state=2)
model = RandomForestRegressor(random_state=32131)

#3. 훈련
model.fit(x_trn, y_trn_org)

#4 평가
result = model.score(x_tst, y_tst_org)

print('점수 :', result)

# LDA 점수 : 0.47374367448542043
# LDA 점수 : 0.48526932479348295
# PCA 점수 : 0.48486823558126624