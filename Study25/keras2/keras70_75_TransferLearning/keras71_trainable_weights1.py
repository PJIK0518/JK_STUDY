import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import random
import tensorflow as tf

RS= 111
random.seed(RS)
np.random.seed(RS)
tf.random.set_seed(RS)

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))


""" model.summary()
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 3)                 6         
                                                                 
 dense_1 (Dense)             (None, 2)                 8         
                                                                 
 dense_2 (Dense)             (None, 1)                 3         
                                                                 
=================================================================
Total params: 17 (68.00 Byte)
Trainable params: 17 (68.00 Byte)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________ """

""" print(model.weights) : 초기가중치 출력
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32,
numpy=array([[0.02689683, 0.3991047 , 0.81457055]],
dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,)
dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>,

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32,
numpy= array([[-0.66334295,  1.0890329 ],
              [-1.0746403 ,  0.7887989 ],
              [-0.20886594, -0.7971348 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,
              
<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32,
numpy = array([[-0.9088995 ],
               [ 0.53127956]], dtype=float32)>,
               <tf.Variable 'dense_2/bias:0'
               shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>] """
# weight : 입력되는 데이터 및 모델 구성에 따라 가중치가 numpy array 형태로 생성 (Input, Output)
# bias   : 최초 bias는 0으로 
print('++++++++++++++++++++++++++++++++++++++++')
""" print(model.trainable_weights)
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32,
    numpy=array([[0.02689683, 0.3991047 , 0.81457055]],dtype=float32)>,
    <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>,
    
    <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
        array([[-0.66334295,  1.0890329 ],
            [-1.0746403 ,  0.7887989 ],
            [-0.20886594, -0.7971348 ]], dtype=float32)>,
            
    <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32,
        numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0'
        shape=(2, 1) dtype=float32,
    numpy= array([[-0.9088995 ],
        [ 0.53127956]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,)
        dtype=float32, numpy=array([0.], dtype=float32)>] """
        
# Random seed를 전체적으로 고정해두면 초기 가중치 고정
# 훈련 가능한 가중치 = 훈련 시 갱신되는 가중치

""" print(len(model.weights))           6 """    
""" print(len(model.trainable_weights)) 6 """
# weight 및 bias의 shape 상관없이 묶여서 Layer별로 weight 하나 bias 하나씩

##### [ Freezen : 동결 ] #####
# 괜찮게 만들어 놓은 모델을 끌어다가 훈련없이 쓰기!!
# 성능 및 시간적 측면에서 이득

# Hugging Face에서 이미 공유중인...
# 가져온걸 그대로 쓴다 : 동결
# 단, 입력 및 출력 데이터는 내가 평가하고 싶은 데이터를 기준으로 바꿔줘야한다

#################### [ 동결 ] ####################

model.trainable = False
""" print(len(model.weights))           6 """    
""" print(len(model.trainable_weights)) 0 """

""" model.summary()
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 3)                 6         
                                                                 
 dense_1 (Dense)             (None, 2)                 8         
                                                                 
 dense_2 (Dense)             (None, 1)                 3         
                                                                 
=================================================================
Total params: 17 (68.00 Byte)
Trainable params: 0 (0.00 Byte)
Non-trainable params: 17 (68.00 Byte)
_________________________________________________________________ """

""" print(model.weights)
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[0.02689683, 0.3991047 , 0.81457055]], dtype=float32)>,
 <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>,
 <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.66334295,  1.0890329 ],
       [-1.0746403 ,  0.7887989 ],
       [-0.20886594, -0.7971348 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,)
dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0'
shape=(2, 1) dtype=float32, numpy=
array([[-0.9088995 ],
       [ 0.53127956]], dtype=float32)>,
<tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>] """

""" print(model.trainable_weights) [] """