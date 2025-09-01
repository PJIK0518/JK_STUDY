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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

x_trn = x_trn.reshape(x_trn.shape[0],-1)
x_tst = x_tst.reshape(x_tst.shape[0],-1)

n = 10
lda = LDA(n_components=n-1)
lda.fit(x_trn,y_trn)
x_trn = lda.transform(x_trn)
x_tst = lda.transform(x_tst)

from tensorflow.keras.utils import to_categorical

# y_trn = to_categorical(y_trn, num_classes=n)
# y_tst = to_categorical(y_tst, num_classes=n)

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

model.add(Dense(1, activation = 'softmax'))


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

[Conv1D]
loss : 4.568159580230713
acc  : 0.04600000008940697
acc  : 0.046
시간 : 67.59857845306396
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

path = './_save/keras40/cifar100/'
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

S = time.time()

hist = model.fit(x_trn, y_trn,
                 epochs = 100,
                 batch_size = 5000,
                 verbose = 1,
                 validation_split = 0.2,
                 callbacks = [ES, MCP],
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

