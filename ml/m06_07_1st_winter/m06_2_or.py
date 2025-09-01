# m06_1.copy

import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,1,1,1]

#2. 모델
# model = Perceptron()
# model = LogisticRegression()
model = LinearSVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가 예측
y_prd = model.predict(x_data)

result = model.score(x_data, y_data)
ACC = accuracy_score(y_data, y_prd)

print(model)
print('model.score :', result)
print('accrc.score :', ACC)

""" Perceptron()
model.score : 1.0
accrc.score : 1.0 """

""" LogisticRegression()
model.score : 0.75
accrc.score : 0.75 """

""" LinearSVC()
model.score : 1.0
accrc.score : 1.0 """