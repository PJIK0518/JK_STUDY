import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# pipeline : 여러가지 ㅈ전처리를 묶어 놓는 툴!
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

# scaler = MinMaxScaler()
# x_trn = scaler.fit_transform(x_trn)
# x_tst = scaler.transform(x_tst)

#2. 모델
# model = RandomForestClassifier()
# model = make_pipeline(StandardScaler(), RandomForestClassifier())
# model = make_pipeline(MinMaxScaler(), SVC())
model = Pipeline([('mms', MinMaxScaler()),
                  ('svc', SVC())])

# 모델이랑 스케일러를 묶어서 모델 구성
# 순서는 스케일러 > 모델

#3. gnsfus
model.fit(x_trn, y_trn)

#4. 평가 예측
scr = model.score(x_tst, y_tst)
print('scr :', scr)

y_prd = model.predict(x_tst)
acc = accuracy_score(y_tst, y_prd)
print('acc :', acc)