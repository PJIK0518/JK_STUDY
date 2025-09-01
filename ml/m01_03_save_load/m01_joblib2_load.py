# m01_1.copy

import numpy as np
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

#1. 데이터

x, y = load_breast_cancer(return_X_y=True)

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              random_state=337,
                                              train_size=0.8,
                                              stratify= y)

# print(x.shape)(569, 30)
# print(y.shape)(569,)

#2. 모델 #3. 훈련 - 불러오기
path = './_save/m01_job/'
model = joblib.load(path + 'm01_joblib_save.joblib')

#4. 평가 예측
result = model.score(x_tst, y_tst)
print(result)

y_prd = model.predict(x_tst)
acc = accuracy_score(y_tst, y_prd)
print(acc)

# joblib.dump(model, path + 'm02_joblib_save.joblib') # 확장자는 하고싶은거 아무거나 해도 가능