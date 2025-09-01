from sklearn.datasets import load_iris

import numpy as np
####################################################################################################
#1. 데이터
####################################################################################################

# datasets = load_iris()

# x = datasets.data
# y = datasets['target']

# print(x.shape) (150, 4)
# print(y.shape) (150,)

x, y = load_iris(return_X_y=True)

####################################################################################################
#2. 모델구성
####################################################################################################
""" Tensorflow 모델 """
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# model = Sequential()

# model.add(Dense(30, activation='relu', input_shape =(4,)))
# model.add(Dense(18))
# model.add(Dense(9))
# model.add(Dense(3, activation='softmax'))

""" 기존 Tensorflow 모델에서 머신 러닝 모델로 전환 """
# 모델 종류 : 회귀 regressor, linear
#            분류 Classfier /// 통상적으로 머신러닝 모델 이름으로 어떤 모델에 적합한지 대충 나옴

""" LinearSVC """
from sklearn.svm import LinearSVC

model = LinearSVC(C = 0.3)

""" LogisticRegression """
from sklearn.linear_model import LogisticRegression # 논리적인 회귀 : 분류모델을 회귀

model = LogisticRegression()

""" DecisionTreeClassifier """
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

""" RandomForestClassifier """
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

Epochs, Batch = (100, 32)

####################################################################################################
#3. 컴파일 훈련
####################################################################################################
""" Tensorflow 모델 """
# model.compile(loss = 'sparse_categorical_crossentropy',
#               optimizer = 'adam',
#               metrics = ['acc'],
#               )

# model.fit(x, y,
#           epochs = Epochs,
#           verbose = 3,
#           )

""" Machine learning """
model.fit(x,y)

####################################################################################################
#4. 평가 예측
####################################################################################################
""" Tensorflow 모델 """
# loss = model.evaluate(x,y)

""" Machine learning """
scor = model.score(x,y)

print(scor)