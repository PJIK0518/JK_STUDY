# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition

##########################################################################
#0. 준비
##########################################################################
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.metrics import F1Score

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

# from tensorflow.keras import mixed_precision
##########################################################################
#1. 데이터
##########################################################################
# mixed_precision.set_global_policy('mixed_float16')

trn_IDG = ImageDataGenerator(rescale=1./255.)
tst_IDG = ImageDataGenerator(rescale=1./255.)


path_trn = 'C:/Study25/_data/kaggle/cat_dog/train_2/'
# path_tst = 'C:/Study25/_data/kaggle/cat_dog/test_2/test/'
S = time.time()
xy_trn = trn_IDG.flow_from_directory(
    path_trn,
    target_size = (200,200),
    batch_size = 100, 
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = True,
    seed = 50,
)   # Found 25000 images belonging to 2 classes

# print(xy_trn[0][0].shape) (100, 200, 200, 3)
# print(xy_trn[0][1].shape) (100,)
# print(len(xy_trn)) 250

# exit()

xy_tst = trn_IDG.flow_from_directory(
    path_trn,
    target_size = (200,200),
    batch_size = 25000,
    class_mode = 'binary',
    color_mode = 'rgb',
    # shuffle = True,
    # seed = 50
)   # Found 25000 images belonging to 2 classes.

# print(xy_trn) <keras.preprocessing.image.DirectoryIterator object at 0x000001B209861100>
# print(len(xy_trn)) 2

# plt.imshow(xy_trn[0][0][0], 'brg')
# print(xy_trn[0][1][0])
# plt.show() 

E = time.time()
print('시간 :', E-S)    # 시간 : 1.5015618801116943


########## Batch의 list화

all_x = []
all_y = []

for i in range(len(xy_trn)):
    x_batch, y_batch = xy_trn[i]
    all_x.append(x_batch)
    all_y.append(y_batch)
E1 = time.time()

# print(type(all_x)) <class 'list'>
# all_x [[100,200,200,3], [100,200,200,3], [100,200,200,3],... [100,200,200,3]]
# 리스트 안에 numpy가 나열, 그냥 바꿔주면 XXXX
print('시간 :', E1-E)   # 시간 : 38.97617244720459

########## list의 numpy화

x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)

E2 = time.time()
print('시간 :', E2-E)

# print(x.shape) (25000, 200, 200, 3) 
# print(y.shape) (25000,)
# print(type(x)) <class 'numpy.ndarray'>
# print(type(y)) <class 'numpy.ndarray'>

exit()

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, train_size=0.85,
                                              shuffle=True,
                                              random_state=42)

# print(x_trn.shape, x_tst.shape, y_trn.shape, y_tst.shape)
# (27, 1024, 1024, 3) (5, 1024, 1024, 3) (27,) (5,)

##########################################################################
#2. 모델 구성
##########################################################################
path_MCP = './_data/kaggle/cat_dog/MCP/'
# model = load_model(path_MCP + 'MCP_0613_2.h5')

model = Sequential()
model.add(Conv2D(40, 5, input_shape=(200,200,3),
                 padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D())

model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

E, B, P, V = (10000, 15, 100, 0.2)

#####################################
### 저장 설정
import datetime

D = datetime.datetime.now()
D = D.strftime('%m%d')
saveNum = f'{D}_0_0.h5'

#####################################
### callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

ES = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=P,
    restore_best_weights=True
)

MCP = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    filepath="".join([path_MCP,'MCP_',saveNum]),
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

S = time.time()
H = model.fit(
    x_trn, y_trn,
    epochs = E,
    batch_size = B,
    verbose = 1,
    validation_split=0.2,
    callbacks=[ES, MCP]
)
E = time.time()
T = E-S

##########################################################################
#4. 평가 예측
##########################################################################
LSAC = model.evaluate(x_tst, y_tst)

#####################################
### 그래프
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(H.history['loss'], color = 'red', label = 'loss')
plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
plt.legend(loc = 'upper right')
plt.grid()

print('loss :', LSAC[0])
print('ACC  :', LSAC[1])
print('time :', T)
print('save :', saveNum)
plt.show()