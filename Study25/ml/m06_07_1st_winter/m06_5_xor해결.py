# m06_4.copy
import warnings

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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
# model = Perceptron()
# model = LogisticRegression()
# model = LinearSVC()
model = Sequential()
model.add(Dense(10, input_dim=2, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

"""
model.score : [0.14311829209327698, 1.0]
accrc.score : 1.0
"""

#3. 컴파일 훈련
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['acc'])

model.fit(x_data, y_data,
          epochs=2000,
          verbose = 3)

#4. 평가 예측
y_prd = model.predict(x_data)
y_prd = np.round(y_prd)

result = model.evaluate(x_data, y_data)
ACC = accuracy_score(y_data, y_prd)

print(model)
print('model.score :', result)
print('accrc.score :', ACC)

""" 단층 Perceptron의
epochs = 100
model.score : [0.7513206601142883, 0.5]
accrc.score : 0.5

epochs = 100000
model.score : [0.6931471824645996, 0.5]
accrc.score : 0.5
"""