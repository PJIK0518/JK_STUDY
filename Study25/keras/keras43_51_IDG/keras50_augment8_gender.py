# 증폭 : 50-1 복사

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
path_NP = 'C:/Study25/_data/kaggle/men_women/'

x = np.load(path_NP + 'x_trn.npy')
y = np.load(path_NP + 'y_trn.npy')

# print(x_trn.shape) (3309, 150, 150, 3)
# print(y_trn.shape) (3309,)

### 전체 데이터 증폭
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
Augment_size = 491


label_0_indices = np.where(y == 0)[0]

randidx = np.random.choice(label_0_indices, size = Augment_size, replace=False)  

x_augmented = x[randidx].copy()

y_augmented = y[randidx].copy()

x_augmented = x_augmented.reshape(Augment_size, 200, 200, 3)

x_augmented, y_augmented = IDG.flow(
    x = x_augmented,
    y = y_augmented,
    batch_size=Augment_size,
    shuffle=False
).next()

x = x.reshape(-1,200, 200, 3)
x = x.reshape(-1,200, 200, 3)

x = np.concatenate([x, x_augmented])
y = np.concatenate([y, y_augmented])

# print(np.unique(y, return_counts=True)) (array([0., 1.], dtype=float32), array([1900, 1900], dtype=int64))

x_trn,x_tst,y_trn,y_tst = train_test_split(x, y,
                                           train_size=0.6,
                                           shuffle=True,
                                           random_state=42)

x_trn = x_trn/255.
x_tst = x_tst/255.

##########################################################################
#2. 모델 구성
##########################################################################
### 모델 불러오기
path_MCP = './_data/kaggle/men_women/MCP/'
path_S = 'C:/Study25/_data/kaggle/men_women/save/'
# model = load_model(path_MCP + 'MCP_0617_0_1.h5')

model = Sequential()
model.add(Conv2D(3, 2, input_shape=(200, 200, 3),
                 padding='same', activation='relu'))
model.add(MaxPool2D(2))
model.add(Dropout(0.4))

model.add(BatchNormalization())
model.add(Conv2D(3, 2, padding='same'))
model.add(MaxPool2D(2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(3, activation='relu'))
model.add(Dropout(0.4))

model.add(BatchNormalization())
model.add(Dense(3, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))

E, B, P, V = (100000, 100, 50, 0.2)

'''
save : 0617_0_1
loss : 0.7082248330116272
acc  : 0.5019736886024475


'''

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
path_W = 'C:/Study25/_data/kaggle/men_women/weights/'
# model.load_weights(path_W + 'save_06_16_0_0.h5')

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
model.save_weights(path_W + f'weights_{saveNum}.h5')

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
plt.title('gender_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(H.history['loss'], color = 'red', label = 'loss')
plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
plt.legend(loc = 'upper right')
plt.grid()

print('save :', saveNum)
print('loss :', LSAC[0])
print('acc  :', LSAC[1])
print('Vlss :', H.history['val_loss'][-1])
print('Vacc :', H.history['val_acc'][-1])
plt.show()