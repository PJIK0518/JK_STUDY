import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import random
import tensorflow as tf

RS= 111
random.seed(RS)
np.random.seed(RS)
tf.random.set_seed(RS)

#1. 데이터
x = np.array([1])
y = np.array([1])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))

############### [ 동 결 ] ###############
model.trainable = True      # 안동결
# model.trainable = False     # 동결

print('🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹')

# print(model.weights)
""" 동결, 안동결 : 훈련전에는 변화ㄴㄴㄴㄴ
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32,
numpy=array([[0.02689683, 0.3991047 , 0.81457055]], dtype=float32)>,
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>,

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.66334295,  1.0890329 ],
[-1.0746403 ,  0.7887989 ],
[-0.20886594, -0.7971348 ]], dtype=float32)>,
<tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,

<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.9088995 ],
[ 0.53127956]], dtype=float32)>,
<tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>] """

#3. 컴파일 훈련
model.compile(loss=  'mse', optimizer ='adam')
model.fit(x, y, epochs = 1000, batch_size = 1)

#4. 평가 예측
x_prd = np.array([1])
y_prd = model.predict(x_prd)

print(y_prd) # [[0.99999994]]

''' 안동결
[[0.99999994]
 [1.7050588 ]
 [2.4101176 ]
 [3.1151764 ]
 [3.8202355 ]]'''
 
''' 동결 : 가중치 갱신이 없다 > epoch를 아무리 늘려도 1에포랑 결과가 같음 
[[0.39851868]
 [0.79703736]
 [1.1955559 ]
 [1.5940747 ]
 [1.9925929 ]]'''

print(model.weights)
""" [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32,
 numpy=array([[0.08203759, 0.4542753 , 0.76692945]], dtype=float32)>,
 
 <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32,
 numpy=array([ 0.05514081,  0.05517091, -0.04764121], dtype=float32)>,
 
 h1-1 = 0.1371784
 h1-2 = 0.50944621
 h1-3 = 0.71928824
 
 <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32,
 numpy=array([[-0.7367257 ,  1.1609179 ],
       [-1.1323979 ,  0.8449857 ],
       [-0.26247048, -0.74500674]], dtype=float32)>,
 <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32,
 numpy=array([-0.05499841,  0.05348743], dtype=float32)>,
 
 h2-1 = -0.10106285276488 -0.576895818366959 -0.1887919296111552 -0.05499841
      = -0.9217490107429942
 h2-2 = 0.15925286005336 + 0.430474762369197 -0.5358745868027376 + 0.05348743
      = 0.1073404656198194
      
 <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32,
 numpy=array([[-0.96777993],
       [ 0.4994912 ]], dtype=float32)>,
 <tf.Variable 'dense_2/bias:0' shape=(1,)
 dtype=float32, numpy=array([0.0543341], dtype=float32)>]
 y = 0.8920501930944242 + 0.0536156179810023 + 0.0543341
   = 0.9999999110754265

 
"""
# 출력한 가중치로 손계산 진행
# 동결
# 훈련