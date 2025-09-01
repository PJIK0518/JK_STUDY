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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

x_trn = x_trn.reshape(x_trn.shape[0],-1)
x_tst = x_tst.reshape(x_tst.shape[0],-1)

n = 2
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

model.add(Dense(n, activation = 'softmax'))


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
# LDA : 0.9105058365758755
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
save : 0623_0_2
loss : 0.6876567006111145
ACC  : 0.8132295719844358
"""
