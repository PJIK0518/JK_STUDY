# LDA : 
# PCA처럼 차원을 축소하는 방식 중 하나  
# y 값도 만져주는 녀석 >> 지도 학습 
#   How? 새로운 축을 x 데이터의 label끼리 나눌 수 있게 생성
    
##### 순서!!!! : split > scale > PCA / trn에 대해서 scaling 밖 범위의 값에 대해 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# discriminant : 판별 /// 선형판별분석

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

for i in range(1, x.shape[1]-1):
    # pca = PCA(n_components=i)
    lda = LDA(n_components=i) # label 값 종류 -1 까지 n_component 
    lda.fit_transform(x_trn, y_trn)
    x_trn_lda = lda.transform(x_trn)
    x_tst_pca = lda.transform(x_tst)
    
    #2. model
    model = RandomForestClassifier(random_state=2)

    #3. 훈련
    model.fit(x_trn_lda, y_trn)

    #4 평가
    y_tst_lda = model.predict(x_tst_pca)
    
    result = model.score(x_tst_pca, y_tst_lda)

    print(i, '의 점수 :', result)
    
# 1 의 점수 : 1.0