## 52_1.copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Flatten

import numpy as np

##########################################################################
#1. 데이터
##########################################################################

datasets = np.array([1,2,3,4,5,6,7,8,9,10])     
      
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])          

y = np.array([4,5,6,7,8,9,10])

x = x.reshape(x.shape[0], x.shape[1], 1)

##########################################################################
#2. 모델 구성
##########################################################################
model = Sequential()
# model.add(SimpleRNN(units=130, input_shape = (3,1), activation='relu'))
# model.add(LSTM(units=130, input_shape = (3,1), activation='relu'))
# model.add(GRU(units=150, input_shape = (3,1), activation='relu'))
model.add(Conv1D(10, kernel_size= 2, input_shape = (3,1),
                 padding = 'same', activation='relu'))
# Output = (None, 2, 10) // padding = valid
        #  (None, 3, 10) // padding = same
 
model.add(Conv1D(9,2, activation='relu'))
# Output = (None, 2, 9)

model.add(Flatten())
# Output = (None, 18)

model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# model.summary()
# exit()

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y,
          epochs = 1000)

##########################################################################
#4. 평가 예측
##########################################################################
loss = model.evaluate(x, y)

x_prd = np.array([8,9,10]).reshape(1,3,1)
y_prd = model.predict(x_prd)

print('lss :', loss)
print('rst :', y_prd)

''' [SimpleRNN 기준]
[ SimpleRNN ]
lss : 1.4025515326920868e-07
rst : [[11.001318]]
[ LSTM ] : 시계열에서는 최강 but 뒤지게 느리다
lss : 1.8311702660867013e-05
rst : [[10.992816]]
[ GRU ]
lss : 0.0007120014051906765
rst : [[10.907407]]
'''

''' [LSTM 기준]
[ SimpleRNN ]
lss : 2.9083145136610256e-07
rst : [[11.00916]]
[ LSTM ]
lss : 2.6687863282859325e-05
rst : [[11.000248]]
[ GRU ]
lss : 0.00039025116711854935
rst : [[10.922967]]
'''

''' [GRU 기준]
[ SimpleRNN ]
lss : 5.970498762053467e-08
rst : [[11.047629]]
[ LSTM ]
lss : 2.146683618775569e-05
rst : [[11.007244]]
[ GRU ]
lss : 3.3151347906823503e-06
rst : [[10.992953]]
'''
