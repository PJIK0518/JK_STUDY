import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# pipeline : 여러가지 ㅈ전처리를 묶어 놓은 툴!
from sklearn.pipeline import make_pipeline, Pipeline

#########################
#1. 데이터

x, y = load_iris(return_X_y=True)

x_trn, x_tst, y_trn, y_tst = train_test_split(
     x, y,
     train_size=0.8,
     shuffle=True,
     random_state=777,
     stratify=y
)

parameters = [
     {'rfc__n_estimators' : [100, 200], 'rfc__max_depth' : [5, 6, 10], 'rfc__min_samples_leaf': [3,10]}, #12
     {'rfc__max_depth' : [6,8,10,12], 'rfc__min_samples_leaf' : [3,5,7,10]}, # 16
     {'rfc__min_samples_leaf' : [3,5,7,9], 'rfc__min_samples_split' : [2,3,5,10]}, #16
     {'rfc__min_samples_split' : [2,3,5,6]}, #4  
] # total : 12+16+16+4 + 1 : 전부다 돌리고 마지막에 최상의 pram으로 한 번 추가
  # Pipeline에 GridSearchCV를 쓰기 위해서는 

#2. 모델
pipe = make_pipeline((MinMaxScaler(), RandomForestClassifier()))
# parametor tuple의 각 파라미터 명을 full name으로 설정필요

model = GridSearchCV(pipe, parameters, cv = 5, verbose=1, n_jobs=-1)

#3. gnsfus
model.fit(x_trn, y_trn)

#4. 평가 예측
scr = model.score(x_tst, y_tst)
print('scr :', scr)

y_prd = model.predict(x_tst)
acc = accuracy_score(y_tst, y_prd)
print('acc :', acc)