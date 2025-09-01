### 39_3.copy

### [실습]
#1. 시간 : vs CNN, CPU vs GPU
#2. 성능 : 기존 모델 능가

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

##########################################################################
#1. 데이터
##########################################################################
(x_trn, y_trn), (x_tst,y_tst) = cifar10.load_data()

# print(x_trn.shape, y_trn.shape) (50000, 32, 32, 3) (50000, 10)
# print(x_tst.shape, y_tst.shape) (10000, 32, 32, 3) (10000, 10)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

x_trn = x_trn.reshape(x_trn.shape[0],-1)
x_tst = x_tst.reshape(x_tst.shape[0],-1)

n = 10
lda = LDA(n_components=n-1)
lda.fit(x_trn,y_trn)
x_trn = lda.transform(x_trn)
x_tst = lda.transform(x_tst)

from tensorflow.keras.utils import to_categorical

y_trn = to_categorical(y_trn, num_classes=n)
y_tst = to_categorical(y_tst, num_classes=n)

from sklearn.model_selection import train_test_split


#####################################
### Scaling
x_trn = x_trn/255.0
x_tst = x_tst/255.0


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
import time

S = time.time()

model = Sequential()
model.add(Dense(64, input_shape=(n-1,)))
model.add(Dropout(0.2))

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(16, activation = 'softmax'))
model.add(Dropout(0.2))

model.add(Dense(10, activation = 'softmax'))


#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['acc'])

model.fit(x_trn, y_trn,
                epochs = 50,
                batch_size = 50,        
                verbose = 1)
#4. 평가,예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
results = np.argmax(results, axis=1)
y_tst_a = np.argmax(y_tst, axis=1)
ACC = accuracy_score(results, y_tst_a)

E = time.time()
T = S-E

print('LDA :', ACC)

# loss : 1.7830915451049805
# acc  : 0.3714999854564667
# acc  : 0.3715
# 시간 : 7.877277135848999

'''
[CNN _ GPU]
loss : 0.6803414821624756
acc  : 0.7702000141143799
acc  : 0.7702

[DNN _ CPU]
loss : 1.3164840936660767
acc  : 0.5637999773025513
acc  : 0.5638
시간 : 541.0520420074463

[DNN _ GPU]
loss : 1.328972339630127
acc  : 0.5616999864578247
acc  : 0.5617
시간 : 729.8132548332214

[LSTM]
loss : 2.2153496742248535
acc  : 0.17159999907016754
acc  : 0.1716
시간 : 72.57322764396667

[Conv1D]
loss : 1.3374377489089966
acc  : 0.5264000296592712
acc  : 0.5264
시간 : 66.23767757415771
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
ES = EarlyStopping(monitor = 'val_acc', mode = 'max',
                   patience= 50, verbose=1,
                   restore_best_weights=True,
)

#####################################
### 파일명
import datetime

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path = './_save/keras40/cifar10/'
filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path,'k40_',date, '_', filename])

#####################################
### MCP
MCP = ModelCheckpoint(monitor = 'val_acc',
                      mode = 'max',
                      save_best_only= True,
                      verbose=1,
                      filepath = filepath,
                      )

#####################################
### 가중치 불러오기
# path_W = './_save/keras40/cifar10/'
# model.load_weights(path_W + 'k39_0611_1741_0045-0.6528.h5')

S = time.time()

hist = model.fit(x_trn, y_trn,
                 epochs = 100,
                 batch_size = 1000,
                 verbose = 3,
                 validation_split = 0.2,
                 callbacks = [ES, MCP])   

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