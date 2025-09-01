import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

## [실습] train_test_split으로 10 : 4: 3으로 나눈다
# x_trn = x[:10]      # [ 1  2  3  4  5  6  7  8  9 10] 
# y_trn = y[:10]      # [ 1  2  3  4  5  6  7  8  9 10]
# x_val = x[10:14]    # [11 12 13 14]
# y_val = y[10:14]    # [11 12 13 14]
# x_tst = x[14:]      # [15 16]
# y_tst = y[14:]      # [15 16]

x_trn,x_tst,y_trn,y_tst = train_test_split(x,y,
                                           train_size=0.8,
                                           shuffle=True,
                                           random_state=42)
'''print(x_trn, x_tst, y_trn, y_tst) // False
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14] [15 16]
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14] [15 16]
'''

x_trn,x_val,y_trn,y_val = train_test_split(x_trn, y_trn,
                                           train_size=0.8,
                                           shuffle=True,
                                           random_state=42)

'''print(x_trn, x_val, x_tst, y_trn, y_val, y_tst) // False
[ 1  2  3  4  5  6  7  8  9 10 11] [12 13 14] [15 16]
[ 1  2  3  4  5  6  7  8  9 10 11] [12 13 14] [15 16]
'''
'''print(x_trn, x_val, x_tst, y_trn, y_val, y_tst) // True
[15  9  8  4  1  5 13  3 12 16] [ 7 10 11 14] [6 2]
[15  9  8  4  1  5 13  3 12 16] [ 7 10 11 14] [6 2]
'''

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
          verbose = 1,
          validation_data=(x_val, y_val))

#4. 평가 예측
L = model.evaluate(x_tst, y_tst)
R = model.predict(n)

print("loss :", L)
print(n,":", R)