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
from tensorflow.keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
import numpy as np
import datetime

##########################################################################
#1. 데이터
##########################################################################
### 이미지 데이터 증폭

path_NP = 'C:/Study25/_data/tensor_cert/horse-or-human/'

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

randidx = np.random.randint(x_trn.shape[0], size = Augment_size)  

x_augmented = x_trn[randidx].copy()

y_augmented = y_trn[randidx].copy()

x_augmented = x_augmented.reshape(Augment_size,150, 150, 3)

x_augmented, y_augmented = IDG.flow(                    # method : class에 종속되는 function
    x = x_augmented,
    y = y_augmented,
    batch_size=Augment_size,
    shuffle=False
).next()

x_trn = x_trn.reshape(-1,150, 150, 3)
x_tst = x_tst.reshape(-1,150, 150, 3)

x_trn = np.concatenate([x_trn, x_augmented])
y_trn = np.concatenate([y_trn, y_augmented])

##########################################################################
#2. 모델 구성
##########################################################################
### 모델 불러오기
path_MCP = './_data/tensor_cert/horse-or-human/MCP/'
path_S = 'C:/Study25/_data/tensor_cert/horse-or-human/save/'
# model = load_model(path_MCP + 'MCP_0613_0_3.h5')

model = Sequential()
model.add(Conv2D(100, 5, input_shape=(150, 150, 3),
                 padding='valid', activation='relu'))
model.add(MaxPool2D(2))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(50, 4, padding='valid'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(10, 3, padding='valid'))
model.add(MaxPool2D(2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(50, activation='sigmoid'))

model.add(Dense(1, activation='sigmoid'))

E, B, P, V = (10000, 15, 50, 0.2)

"""
save : 0616_0_0
loss : 0.022970128804445267
ACC  : 0.9961089491844177

save : 0616_0_1
loss : 0.029441438615322113
ACC  : 0.9922178988326849

save : 0616_0_2
loss : 0.06290081888437271
ACC  : 0.9727626459143969

save : 0617_0_2
loss : 0.04360080510377884
ACC  : 0.9844357976653697

"""

#####################################
### 저장 설정

D = datetime.datetime.now()
D = D.strftime('%m%d')
saveNum = f'{D}_0_2'

#####################################
### callbacks

ES = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=P,
    restore_best_weights=True
)

MCP = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    filepath="".join([path_MCP,'MCP_',saveNum,'.h5']),
    save_best_only=True
)

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
path_W = 'C:/Study25/_data/tensor_cert/horse-or-human/weights/'
# model.load_weights(path_W + f'weight_0616_0_0.h5')

H = model.fit(
    x_trn, y_trn,
    epochs = E,
    batch_size = B,
    verbose = 1,
    validation_split=0.2,
    callbacks=[ES, MCP]
)

#####################################
### 모델 및 가중치 저장
model.save(path_S + f'save_{saveNum}.h5')
model.save_weights(path_W + f'weight_{saveNum}.h5')

##########################################################################
#4. 평가 예측
##########################################################################
LSAC = model.evaluate(x_tst, y_tst)

y_pred = np.round(model.predict(x_tst))

ACC = accuracy_score(y_tst,y_pred)

#####################################
### 그래프
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.title('human_horse')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(H.history['loss'], color = 'red', label = 'loss')
plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
plt.legend(loc = 'upper right')
plt.grid()

print('save :', saveNum)
print('loss :', LSAC[0])
print('ACC  :', ACC)
plt.show()