#16-3.copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_trn,x_tst,y_trn,y_tst = train_test_split(x,y,
                                           train_size=0.8,
                                           shuffle=True,
                                           random_state=42)

# x_trn,x_val,y_trn,y_val = train_test_split(x_trn, y_trn,
#                                            train_size=0.8,
#                                            shuffle=True,
#                                            random_state=42)
# >> 두번짜르지 않고 fit에서 진행도 가능

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(1))

E = 100
B = 1
n = [17]

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_trn, y_trn,
          epochs = E,
          batch_size = B,
          verbose = 2,
          validation_split=0.2)

#4. 평가 예측
L = model.evaluate(x_tst, y_tst)
R = model.predict(n)

print("loss :", L)
print(n,":", R)