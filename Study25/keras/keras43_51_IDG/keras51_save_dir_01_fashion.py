# 수치화해서 변환 및 증폭한 데이터를 이미지 파일로 저장
# 50-1.copy

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
import numpy as np

IDG = ImageDataGenerator(
    # rescale=1./255.,
    horizontal_flip=True,   # 좌우 반전
    # vertical_flip=True,
    width_shift_range=0.1,
    # height_shift_range=0.1,
    rotation_range=15,
    # zoom_range=0.2,
    # shear_range=0.7,
    fill_mode='nearest' 
)
Augment_size = 40000

(x_trn, y_trn), (x_tst, y_tst) = fashion_mnist.load_data()

x_trn = x_trn/255.
x_tst = x_tst/255.

randidx = np.random.randint(x_trn.shape[0], size = Augment_size)  # np.random.randint(60000, 40000) 이랑 동일

# print(randidx)  [23443  7932  7074 ... 56175 17121  6281]
# print(np.max(randidx))  59999
# print(np.min(randidx))  0

x_augmented = x_trn[randidx].copy()     # 메모리 공유로 인한 x_trn과 x_augmented 서로 영향을 줄 수 있기 때문에
                                        # 4만개의 데이터 copy 후 새로운 메모리에 할당 >> 서로 영향 X
y_augmented = y_trn[randidx].copy()     # 동일한 40000개의 배열을 가지는 y_를 
                                        # 40000개의 데이터를 복사
# print(x_augmented)
# print(x_augmented.shape)  (40000, 28, 28)
# print(y_augmented.shape)  (40000,)

x_augmented = x_augmented.reshape(40000,28,28,1)
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],1)   # 두 개는 동일, 이렇게 하면 다른 데이터에도 활용 가능

# print(x_augmented.shape) (40000, 28, 28, 1)

x_augmented, y_augmented = IDG.flow(                # 데이터가 변환 및 증폭되는 설정
    x = x_augmented,
    y = y_augmented,
    batch_size=Augment_size,
    shuffle=False,          # x, y를 같이 넣어줬으니까 섞어도 되지만, x만 넣었다면 섞으면 안됨
    save_to_dir='c:/Study25/_data/_save_img/01_fashion/',
).next()

exit()

# print(x_augmented.shape) (40000, 28, 28, 1)
# print(x_trn.shape) (60000, 28, 28)

x_trn = x_trn.reshape(60000,28,28,1)
x_tst = x_tst.reshape(-1,28,28,1)

# print(x_trn.shape)
# print(x_tst.shape)

x_trn = np.concatenate([x_trn, x_augmented])
y_trn = np.concatenate([y_trn, y_augmented])
#####################################
### OneHotEncoder
y_trn = pd.get_dummies(y_trn)
y_tst = pd.get_dummies(y_tst)

# print(x_trn.shape) (100000, 28, 28, 1)
# print(x_tst.shape) (10000, 28, 28, 1)

##########################################################################
#2. 모델 구성
### 모델 불러오기
path = './_save/keras39/fashion/'
model = load_model(path + 'k39_0612_0903.h5')
#####################################

# model = Sequential()
# model.add(Conv2D(100, 2, input_shape=(28,28,1), activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=80, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=60, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=40, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=20, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(MaxPool2D(4))
# model.add(Dropout(0.2))

# model.add(Flatten())                                           

# model.add(BatchNormalization())
# model.add(Dense(units=100, activation='relu'))                  
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Dense(units=80, activation='relu'))                  
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Dense(units=50, activation='relu'))                  
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Dense(units=20, input_shape=(16,), activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(units=1, activation='softmax'))

##########################################################################
#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc']
              )


# ES = EarlyStopping(monitor = 'val_acc', mode = 'max',
#                    patience= 50, verbose=1,                         # ES에서의 verbose = early stopping 지점을 알 수 있다
#                    restore_best_weights=True,
    
# )
################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
# import datetime

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')              

### 모델 정보 넣기 (epoch, weight)
### 파일 저장

# filename = '{epoch:04d}-{val_loss:.4f}.h5' # 04d : 네자리 수 /// .f4 : 소수넷째자리
# filepath = "".join([path,'k50_',date, '.h5'])

# MCP = ModelCheckpoint(monitor='val_acc',
#                       mode='auto',
#                       save_best_only=True,
#                       verbose = 1,
#                       filepath= filepath # 확장자의 경우 h5랑 같음
#                                          # patience 만큼 지나기전 최저 갱신 지점        
#                       )

# Start = time.time()
# hist = model.fit(x_trn, y_trn,
#                  epochs = 5000,
#                  batch_size = 50,        # CNN에서 batch_size : 그림을 몇 장씩 한번에 처리하냐
#                  verbose = 3,
#                  validation_split = 0.2,
#                  callbacks = [ES, MCP])                    # 두 개 이상을 불러올수도 있다
# End = time.time()

# T = End - Start

#4. 평가,예측
loss = model.evaluate(x_tst, y_tst,
                      verbose= 1)
results = model.predict(x_tst)
results = np.argmax(results, axis=1)
y_tst = np.argmax(y_tst.values, axis=1)
ACC = accuracy_score(results, y_tst)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss[0])    # loss, categorical_crossentropy
print("acc  :", loss[1])    # metrics, accuracy
print("acc  :", ACC)
# print("시간 :", T)

'''
loss : 0.22302168607711792
acc  : 0.9302999973297119
acc  : 0.9303
시간 : 1882.5303266048431

loss : 0.20780806243419647
acc  : 0.9316999912261963
acc  : 0.9317
시간 : 1904.9725253582

[augment]
loss : 1.138975977897644
acc  : 0.6726999878883362
acc  : 0.6727
'''