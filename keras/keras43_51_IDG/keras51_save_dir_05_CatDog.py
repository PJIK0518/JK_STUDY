# 증폭 : 50-1 복사

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

import matplotlib.pyplot as plt
import numpy as np
import datetime

##########################################################################
#1. 데이터
##########################################################################
### 이미지 데이터 증폭
path_NP = 'C:/Study25/_data/_save_npy/'

x = np.load(path_NP + 'keras44_01_x_test_64.npy')
y = np.load(path_NP + 'keras44_01_y_train_0.npy')

x_trn,x_tst,y_trn,y_tst = train_test_split(x, y,
                                           train_size=0.6,
                                           shuffle=True,
                                           random_state=50,
                                           stratify=y,
                                           )

IDG = ImageDataGenerator(
    # rescale=1./255.,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest' 
)
Augment_size = 40000

x_trn = x_trn/255.
x_tst = x_tst/255.

randidx = np.random.randint(x_trn.shape[0], size = Augment_size)  

x_augmented = x_trn[randidx].copy()

y_augmented = y_trn[randidx].copy()

x_augmented = x_augmented.reshape(Augment_size,
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],
                                  3)

x_augmented, y_augmented = IDG.flow(
    x = x_augmented,
    y = y_augmented,
    batch_size=Augment_size,
    shuffle=False,
    save_to_dir='c:/Study25/_data/_save_img/05_CatDog/',
).next()

exit()

x_trn = x_trn.reshape(-1,
                      x_augmented.shape[1],
                      x_augmented.shape[2],
                      x_augmented.shape[3])

x_tst = x_tst.reshape(-1,
                      x_augmented.shape[1],
                      x_augmented.shape[2],
                      x_augmented.shape[3])

x_trn = np.concatenate([x_trn, x_augmented])
y_trn = np.concatenate([y_trn, y_augmented])

##########################################################################
#2. 모델 구성
##########################################################################
### 모델 불러오기
path_MCP = './_data/kaggle/cat_dog/MCP/'
path_S = 'C:/Study25/_data/kaggle/cat_dog/save/'
model = load_model(path_S + 'save_0617_0_5.h5')

# model = Sequential()
# model.add(Conv2D(50, 3, input_shape=(32,32,3),
#                  padding='valid', activation='relu'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(20, 3, padding='valid'))
# model.add(MaxPool2D(2))
# model.add(BatchNormalization())

# model.add(Activation('relu'))
# model.add(Dropout(0.2))

# model.add(Flatten())

# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Dense(10, activation='sigmoid'))

# model.add(Dense(1, activation='sigmoid'))
# E, B, P, V = (10000, 200, 100, 0.2)

#####################################
### 저장 설정

# D = datetime.datetime.now()
# D = D.strftime('%m%d')
# saveNum = f'{D}_0_5'

#####################################
### callbacks

# ES = EarlyStopping(
#     monitor='val_acc',
#     mode='max',
#     patience=P,
#     restore_best_weights=True
# )

# MCP = ModelCheckpoint(
#     monitor='val_acc',
#     mode='max',
#     filepath="".join([path_MCP,'MCP_',saveNum,'.h5']),
#     save_best_only=True
# )

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics=['acc']
)

#####################################
### 가중치 불러오기
# path_W = 'C:/Study25/_data/kaggle/cat_dog/weights/'
# model.load_weights(path_W + 'weights_0615_3_0.h5')


# S = time.time()
# H = model.fit(
#     x_trn, y_trn,
#     epochs = E,
#     batch_size = B,
#     verbose = 1,
#     validation_split=0.2,
#     callbacks=[ES, MCP]
# )
# E = time.time()
# T = E-S

#####################################
### 모델 및 가중치 저장
# model.save(path_S + f'save_{saveNum}.h5')
# model.save_weights(path_W + f'weights_{saveNum}.h5')

##########################################################################
#4. 평가 예측
##########################################################################
LSAC = model.evaluate(x_tst, y_tst)

y_pred = np.round(model.predict(x_tst))

ACC = accuracy_score(y_tst, y_pred)

#####################################
### 그래프
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9, 6))
# plt.title('loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(H.history['loss'], color = 'red', label = 'loss')
# plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
# plt.legend(loc = 'upper right')
# plt.grid()

# print('save :', saveNum)
print('loss :', LSAC[0])
print('acc  :', LSAC[1])
# print('Vlss :', np.round(H.history['val_loss'][-1], 6))
# print('Vacc :', np.round(H.history['val_acc'][-1], 6))
# print('time :', T)
# plt.show()


'''
loss : 0.5776981115341187
acc  : 0.697700023651123
'''