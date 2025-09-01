### coy. 43-1
### brain

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Dropout, MaxPool2D, Flatten

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

##########################################################################
#1. 데이터
##########################################################################
trn_IDG = ImageDataGenerator(
    rescale=1./255.,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest' 
)                                   # 증폭 : 너무 과하면 데이터가 오염 /
                                    #        하더라도 원본은 유지하고 해야한다
                                    
tst_IDG = ImageDataGenerator(       # 테스트 데이터의 경우에 수정 없이 모델에 입력되는 데이터
    rescale=1./255.,                # 수치화만 진행
)

# 파일 데이터 경로 설정
path_trn = './_data/image/brain/train/'
path_tst = './_data/image/brain/test/'
                                    # 현재 train 폴더 안에 있는 폴더별로 인식 가능
                                    
xy_trn = trn_IDG.flow_from_directory(        # 특정 폴더에서 가져와서 인스턴스화된 기능을 먹여라
    path_trn,                                # 경로
    target_size=(200, 200),                  # Resize, 모든 데이터를 동일한 사이즈로 규격화
    batch_size=1000,                         # 총 데이터 : (160,200,200,1) > 16*(10,200,200,1)
    class_mode='binary',                     # 이진분류 // 다중분류 : categorical
    color_mode='grayscale',                  
    shuffle=True,
    seed=30,                                # ImageDataGenerator_shuffle의 random_state
)                                            # Found 160 images belonging to 2 classes

xy_tst = tst_IDG.flow_from_directory(        
    path_tst,                                
    target_size=(200, 200),                 
    batch_size=1000,          
    class_mode='binary',          
    color_mode='grayscale',
    # shuffle=True,                          # Test는 꼭 안해도 됨
)                                            # Found 120 images belonging to 2 classes

x_trn = xy_trn[0][0]
y_trn = xy_trn[0][1]

x_tst = xy_tst[0][0]
y_tst = xy_tst[0][1]

# n = 4
# plt.imshow(x_trn[n], 'gray')
# print(y_trn[n])
# plt.show()
# exit()


# print(x_trn.shape, y_trn.shape) (160, 200, 200, 1) (160,)
# print(x_tst.shape, y_tst.shape) (120, 200, 200, 1) (120,)
##########################################################################
#2. 모델 구성
##########################################################################
path_MCP = './_save/keras43/'
# model = load_model(path_MCP + 'MCP_0613_2.h5')

model = Sequential()
model.add(Conv2D(40, 5, input_shape=(200, 200, 1),
                 padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D())

model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

E, B, P, V = (10000, 15, 100, 0.2)

'''
loss : 0.05598684400320053
ACC  : 0.9916666746139526
time : 63.7128210067749
save : 0613_3.h5
'''

#####################################
### 저장 설정
import datetime

D = datetime.datetime.now()
D = D.strftime('%m%d')
saveNum = f'{D}_4.h5'

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

import time

# model.load_weights(path_MCP + 'MCP_0613_2.h5')

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



