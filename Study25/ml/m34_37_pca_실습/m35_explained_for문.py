# PCA : Principal Component Analysis, 주성분분석
    # 컬럼을 새로 만드는 방식 (like. Embedding)
    # How. 데이터의 분포를 기준으로 축을 다시 잡는 것 : 차원의 축소
    # But. 기존 컬럼이 많으면 여러가지 하나의 축으로는 PCA 불가능
    #       > 여러개의 축을 그어서 축 만큼 컬럼(차원)추가
    
##### 순서!!!! : split > scale > PCA / trn에 대해서 scaling 밖 범위의 값에 대해 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
# decomposition : 분해!

#1. 데이터

DS = load_iris()
x = DS.data
y = DS.target

# print(x.shape) (150, 4)
# print(y.shape) (150,)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42
)

### PCA 이전 Scaling??
Scaler = StandardScaler()
x_trn = Scaler.fit_transform(x_trn)
x_tst = Scaler.transform(x_tst)

# 선생님 스타일~
for i in range(x.shape[1]):
    pca = PCA(n_components=i+1) 
    x_trn_pca = pca.fit_transform(x_trn)
    x_tst_pca = pca.transform(x_tst)
    
    #2. model
    model = RandomForestClassifier(random_state=333)

    #3. 훈련
    model.fit(x_trn_pca, y_trn)

    #4 평가
    result = model.score(x_tst_pca, y_tst)

    print(x_trn_pca.shape, '의 점수 :', result)

# 1 의 점수 : 0.9333333333333333
# 2 의 점수 : 0.9555555555555556
# 3 의 점수 : 0.9555555555555556
# 4 의 점수 : 1.0

evr = pca.explained_variance_ratio_ # 설명가능한 변화율
print('evr        :', evr)       
# evr   : [0.7070102  0.24507687 0.04266747 0.00524546]
print('evr_s      :', sum(evr))  
# evr_s : 1.0 >> PCA가 전체 columns를 활용한 정도?

# Cumulative Sum, 누적 합
evr_cumsum = np.cumsum(evr)
print('evr_cumsum :', evr_cumsum)
# evr_cumsum : [0.7070102  0.95208707 0.99475454 1.   ]

# 시각화
import matplotlib.pyplot as plt

plt.plot(evr_cumsum)
plt.grid()
plt.show()