# 증폭 : 50-6 복사

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
import numpy as np
import datetime

##########################################################################
#1. 데이터
##########################################################################
### 이미지 데이터 증폭

path_NP = './Study25/_data/tensor_cert/horse-or-human/'

x = np.load(path_NP + 'x_trn.npy')
y = np.load(path_NP + 'y_trn.npy')

# print(x.shape) (1027, 150, 150, 3)
# print(y.shape) (1027,)

x_trn,x_tst,y_trn,y_tst = train_test_split(x, y,
                                           train_size=0.75,
                                           shuffle=True,
                                           random_state=42)
IDG = ImageDataGenerator(
    # rescale=1./255.,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest' 
)

Augment_size = 500

x_trn = x_trn/255.
x_tst = x_tst/255.

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
              input_shape = (150,150,3))

vgg16.trainable = False

model = Sequential()

model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation = 'softmax'))

### 실습
# [FLATTEN]
# 1. 전이학습 vs 기존최고기록 (acc  : 0.7702)
# 2. vgg 16 가중치 동결
        # loss : 0.6933310031890869
        # ACC  : 0.5019455252918288

# 3. vgg 16 가중치 안동결
        # loss : 0.6942299604415894
        # ACC  : 0.5019455252918288

# [GAP]
# 1. 전이학습 vs 기존최고기록 (acc  : 0.7702)
# 2. vgg 16 가중치 동결
        # loss : 0.6955475807189941
        # ACC  : 0.5019455252918288

# 3. vgg 16 가중치 안동결
        # loss : 0.6935269236564636
        # ACC  : 0.5019455252918288
"""
[Conv2D]
save : 0616_0_0
loss : 0.022970128804445267
ACC  : 0.9961089491844177

[LSTM]
save : 0623_0_2
loss : 0.7139459848403931
ACC  : 0.4980544747081712

[Conv1D]
"""

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics=['acc']
)

H = model.fit(
    x_trn, y_trn,
    epochs = 10,
    batch_size = 150,
    verbose = 1,
)

##########################################################################
#4. 평가 예측
##########################################################################
LSAC = model.evaluate(x_tst, y_tst)

y_pred = np.round(model.predict(x_tst))

ACC = accuracy_score(y_tst,y_pred)

#####################################
### 그래프
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9, 6))
# plt.title('human_horse')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(H.history['loss'], color = 'red', label = 'loss')
# plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
# plt.legend(loc = 'upper right')
# plt.grid()

print('loss :', LSAC[0])
print('ACC  :', ACC)
# plt.show()