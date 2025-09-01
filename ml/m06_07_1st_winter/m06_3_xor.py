# m06_2.copy

import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0,1,1,0])

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
model.score : 0.5
accrc.score : 0.5 """

""" LogisticRegression()
model.score : 0.5
accrc.score : 0.5 """

""" LinearSVC()
model.score : 0.5
accrc.score : 0.5 """

### XOR 모델의 경우 1969~1986년 까지 해결을 못함 
### 인공지능의 제 1차 겨울 발생 > 다중 레이어