# m06_5.copy

import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


### 초기 랜덤값 고정!!
RS = 42
''' tensorflow random 값 고정 '''
import tensorflow as tf
tf.random.set_seed(RS)
''' numpy random 값 고정 '''
import numpy as np
np.random.seed(RS)
''' python random 값 고정 '''
import random
random.seed(RS)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,1,1,0]

#2. 모델
# model = Perceptron()               # 미해결
# model = LogisticRegression()       # 미해결
# model = LinearSVC()                # 미해결
# model = SVC()                      # 해결
model = DecisionTreeClassifier()   # 해결


#3. 훈련
model.fit(x_data, y_data)

#4. 평가 예측
y_prd = model.predict(x_data)
y_prd = np.round(y_prd)

result = model.score(x_data, y_data)
ACC = accuracy_score(y_data, y_prd)

print(model)
print('model.score :', result)
print('accrc.score :', ACC)

""" SVC()
model.score : 1.0
accrc.score : 1.0 """

""" DecisionTreeClassifier()
model.score : 1.0
accrc.score : 1.0 """