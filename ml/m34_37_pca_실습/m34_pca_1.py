# PCA : Principal Component Analysis, 주성분분석 (원래는 model의 layer로 개발 됐지만 지금은 그냥 데이터 처리에 사용)
    # 컬럼을 새로 만드는 방식 (like. Embedding)
    # How. 데이터의 분포를 기준으로 축을 다시 잡는 것 : 차원의 축소
    # But. 기존 컬럼이 많으면 여러가지 하나의 축으로는 PCA 불가능
    #       > 여러개의 축을 그어서 축 만큼 컬럼(차원)추가
    #       > y값이 없기 때문에 비지도 학습

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
# decomposition : 분해!

#1. 데이터

DS = load_iris()
x = DS.data
y = DS.target

# print(x.shape) (150, 4)
# print(y.shape) (150,)

### PCA 이전 Scaling??
# Scaler = StandardScaler()
# x = Scaler.fit_transform(x)

# PCA = PCA(n_components=3) 
# x = PCA.fit_transform(x)

# print(x)
# print(x.shape)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42
)

#2. model
model = RandomForestClassifier(random_state=333)

#3. 훈련

model.fit(x_trn, y_trn)

#4 평가

result = model.score(x_tst, y_tst)

print(x.shape, '의 점수 :', result)
