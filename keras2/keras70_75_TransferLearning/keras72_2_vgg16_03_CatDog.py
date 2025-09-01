# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition
# 45.copy
##########################################################################
#0. 준비
##########################################################################
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

RS = 518
##########################################################################
#1. 데이터
##########################################################################
path_NP = './Study25/_data/kaggle/cat_dog/'

x = np.load(path_NP + '04_x_train.npy')
y = np.load(path_NP + '04_y_train.npy')

x_trn,x_tst,y_trn,y_tst = train_test_split(x, y,
                                           train_size=0.6,
                                           shuffle=True,
                                           random_state=50,
                                           stratify=y,
                                           )

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

vgg16.trainable = True

model = Sequential()

model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation = 'softmax'))

### 실습
# [FLATTEN]
# 1. 전이학습 vs 기존최고기록 (0.5257750153541565)
# 2. vgg 16 가중치 동결
    # 0.5
    # 211.863831281662
# 3. vgg 16 가중치 안동결
    # 0.5
    # 590.0822637081146
        
# [GAP]
# 2. vgg 16 가중치 동결
    # 0.5
    # 212.91225671768188
# 3. vgg 16 가중치 안동결
    # 0.5
    # 684.8496518135071

'''
[LSTM]
save : 0623_0_5
loss : 0.6918511986732483
acc  : 0.5257750153541565
'''

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics=['acc']
)

S = time.time()
H = model.fit(
    x_trn, y_trn,
    epochs = 10,
    batch_size = 5000,
    verbose = 1,
    validation_split=0.2,
)
E = time.time()
T = E-S

##########################################################################
#4. 평가 예측
##########################################################################
LSAC = model.evaluate(x_tst, y_tst)

y_pred = np.round(model.predict(x_tst))

ACC = accuracy_score(y_tst, y_pred)

print(ACC)
print(T)