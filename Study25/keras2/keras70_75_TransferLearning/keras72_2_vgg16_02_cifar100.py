### 40_4.copy

### [실습]
#1. 시간 : vs CNN, CPU vs GPU
#2. 성능 : 기존 모델 능가

from tensorflow.keras.datasets import cifar100

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

##########################################################################
#1. 데이터
##########################################################################
(x_trn, y_trn), (x_tst,y_tst) = cifar100.load_data()

#####################################
### y OneHot
One = OneHotEncoder()

y_trn = y_trn.reshape(-1,1)
y_tst = y_tst.reshape(-1,1)

One.fit(y_trn)

y_trn = One.transform(y_trn).toarray()
y_tst = One.transform(y_tst).toarray()

#####################################
### Scaling
x_trn = (x_trn-127.5)/(510.0)
x_tst = (x_tst-127.5)/(510.0)

##########################################################################
#2. 모델 구성
##########################################################################
### 랜덤고정
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import numpy as np
import random
RS = 111
random.seed(RS)
np.random.seed(RS)
tf.random.set_seed(RS)


from tensorflow.keras.applications import VGG16

vgg16 = VGG16(include_top = False,
              input_shape = (32, 32, 3))

vgg16.trainable = False

model = Sequential()

model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation = 'softmax'))

### 실습
# [FLATTEN]
# 1. 전이학습 vs 기존최고기록 (acc  : 0.5222)
# 2. vgg 16 가중치 동결
# acc  : 0.2253
# 시간 : 175.53777813911438
# 3. vgg 16 가중치 안동결
# acc  : 0.032
# 시간 : 582.7085916996002

# [GAP]
# 1. 전이학습 vs 기존최고기록 (acc  : 0.7702)
# 2. vgg 16 가중치 동결
# acc  : 0.2253
# 시간 : 176.74399065971375

# 3. vgg 16 가중치 안동결
# acc  : 0.032
# 시간 : 598.8027873039246
'''
[CNN-GPU]
loss : 1.812417984008789
acc  : 0.5221999883651733
acc  : 0.5222
시간 : 5364.959238290787

[DNN-CPU]
loss : 3.081491708755493
acc  : 0.2833000123500824
acc  : 0.2833
시간 : 577.3938059806824

[DNN-GPU]
loss : 3.1438710689544678
acc  : 0.28769999742507935
acc  : 0.2877
시간 : 701.1971230506897

[LSTM]
loss : 4.1187052726745605
acc  : 0.07620000094175339
acc  : 0.0762
시간 : 127.99137330055237
'''


##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc']
              )

#####################################
### ES
ES = EarlyStopping(monitor = 'acc', mode = 'max',
                   patience= 50, verbose=1,
                   restore_best_weights=True,
)


S = time.time()

hist = model.fit(x_trn, y_trn,
                 epochs = 10,
                 batch_size = 5000,
                 verbose = 1,
                 callbacks = [ES],
                 )   

E = time.time()

T = E - S

##########################################################################
#4. 평가,예측
##########################################################################
loss = model.evaluate(x_tst, y_tst,
                      verbose= 1)
results = model.predict(x_tst)

#####################################
### 결과값 처리
results = np.argmax(results, axis=1)
y_tst = np.argmax(y_tst, axis=1)
ACC = accuracy_score(results, y_tst)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss[0])
print("acc  :", loss[1])
print("acc  :", ACC)
print("시간 :", T)

