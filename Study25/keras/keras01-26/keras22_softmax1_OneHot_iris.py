# load_iris : 꽃의 특성으로 iris라는 꽃의 품종 예측하는 데이터

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

#1. 데이터
DS = load_iris()

x = DS.data
y = DS.target

# print(x.shape, y.shape) (150, 4) (150,)
'''print(y) (np.unique(y, return_counts=True)) array([0, 1, 2])
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2]
'''
# 데이터가 순차적으로 정리되어 있을 때, 순차적으로 trs, tst를 잘라버리면 특정 범위가 학습 부족
# print(np.unique(y, return_counts=True)) array([0, 1, 2]), array([50, 50, 50]
# print(pd.value_counts(y))
# print(pd.DataFrame(y).value_counts())

# OneHotEncoding
# #1. sklearn
from sklearn.preprocessing import OneHotEncoder
# import numpy as np
# y = y.reshape(-1, 1)            # 기존 y 는 (150, )의 벡터형태 > (150,1)의 행렬형태 // 순서와 값은 바뀌지 않고 형태만 변경
#                                 # numpy는 metrics 형태의 데이터 처리 가능
#                                 # 행렬 > 텐서도 가능, ex. (10,10) > (10,5,2) 등으로 가능
# one = OneHotEncoder()
# one.fit(y)
# y = one.transform(y)

# print(y)
# print(y.shape) # (150, 3)
# print(type(y)) # <class 'scipy.sparse.csr.csr_matrix'>

# y = y.toarray()

# print(y)
# print(y.shape) # (150, 3)
# print(type(y)) # <class 'numpy.ndarray'>
#                # 데이터 형태가 scipy 형태의 데이터라서 모델에 넣기 위해 numpy 형태로 전환 
#                # : .toarray

y = y.reshape(-1, 1)
one = OneHotEncoder(sparse=False) # scipy 형태로 안 만들고, 진행 = to.array
one.fit(y)
y = one.transform(y)

### 데이터가 0,1,2 가아니라 2,3,7 느낌으로 될수도
  # 모든 데이터는 받고 나서 shape랑 return_counts, value_counts를 반드시 해야한다


# #2. pandas
# import pandas as pd
# y = pd.get_dummies(y)

#3. keras
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

# print(y.shape)  (150, 3)      # 다중분류에서는 output layer node의 갯수는 y의 종류 수 와 같다

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size=0.9,
                                              shuffle=True,
                                              random_state=55)

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(3, activation="softmax"))   # activation="linear" : 다중분류인데 선형구조 > 분류 불가
                                            # 0, 1, 2 가 숫자로의 의미로 해석됨 > 품종2가 2개가 되면 품종3으로 해석
                                            # 0, 1, 2 는 수치적인 가치가 아니라 고유한 의미를 가지는 결과값
                                            # : linear 한 모델로는 불가
                                            # : OneHotEncoding, 결과값을 수치가 아니라 위치로 인식하게 변환
                                            # : 데이터 전처리과정, 통상적으로 y 값에만 적용
                                            #
                                            # 0 > 1 0 0 : 위치 = 0번째 컬럼 / 총 수치 = 1
                                            # 1 > 0 1 0 : 위치 = 1번째 컬럼 / 총 수치 = 1
                                            # 2 > 0 0 1 : 위치 = 2번째 컬럼 / 총 수치 = 1
                                            # : 메트리스를 만들어서 같은 숫자로 위치만 표기

E = 100000
B = 15
V = 0.2
P = 10

ES = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience= P,
                   restore_best_weights=True)

#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc'])

#### 다중분류의 특징!!!
 #1. OneHotEncoder
 #2. y label 수 = out put layer node의 수
 #3. activation = 'softmax'
 #4. loss = 'categorical_crossentropy'

S_time = time.time()

H = model.fit(x_trn, y_trn,
          epochs = E,
          batch_size= B,
          verbose = 3,
          validation_split = 0.1,
          callbacks= [ES])

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.title('IRIS')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(H.history['loss'], color = 'red', label = 'loss')
plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
plt.legend(loc = 'upper right')
plt.grid()

plt.figure(figsize=(9,6))
plt.title('IRIS')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.plot(H.history['acc'], color = 'red', label = 'acc')
plt.plot(H.history['val_acc'], color = 'green', label = 'val_acc')
plt.legend(loc = 'lower right')
plt.grid()

E_time = time.time()

time = E_time - S_time

#4. 평가 예측
LSAC = model.evaluate(x_tst, y_tst)

from sklearn.metrics import accuracy_score

R = model.predict(x_tst)      # 다중분류의 결과값을 Acc를 측정하려면 OneHot을 풀어줘서 벡터형으로 바꿔줘야함
                              # pandas, numpy 형태를 통일 시켜줘야함

R = np.argmax(R, axis=1)
# print(R) [0 0 0 2 2 0 2 2 0 0 0 1 2 0 2]
# print(R.shape) (15,)
""" print(y_tst)
[[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]] """
# print(y_tst.shape) (15, 3)

y_tst = np.argmax(y_tst, axis=1)
# print(y_tst) [0 0 0 2 2 0 2 2 0 0 0 1 2 0 2]
# print(y_tst.shape) (15,)

ACC = accuracy_score(y_tst, R)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ') 
print('loss:', LSAC[0]) 
print('acc :', LSAC[1])
print('acc :', ACC)
print('time:', time, '초')
print(R)

plt.show()

'''
loss: 0.01758553273975849
acc : 1.0
time: 9.677772998809814 초
'''