# 36_1.copy

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
                                           # 2D 데이터에 대한 Convolution layer 제작 class

# 수업 : (N,5,5,1)의 이미지를 2*2로 convoltion
# 이미지는 기본적으로 4차원 데이터, 개수, 가로, 세로, 색
                                                          # (  가로,  세로,     색깔)
model = Sequential()                                      # (height,width, channels)
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1))) # 출력 > (4,4,10) : 상위 레이어의 filter는 하위 레이어에서 channel로 들어간다
model.add(Conv2D(5, (2,2))) # kernel_size 만큼 가중치가 각 연산 마다 생성, kernel = weight
model.add(Flatten())        # 다차원 데이터를 2차원을 눌러준다, Param은 없다..!
model.add(Dense(units=10))  # units = output node의 수 / input_shape = (batch, input_dim)
model.add(Dense(10))        # 상위 레이어의 unit은 input_dim
        # Dense는 들어오는 데이터 차원 그대로 유지하면서 연산!
        # 필요한게 Matrix형 데이터? > reshape 필요! (reshape의 기본 : 수치와 순서는 변하면 안된다!)
        # model 안에서도 가능! > 원하는 layer에 맞게 reshape해서 가능
        # 이미지를 보고 객체가 뭔지 분류 해야한다 >> matrix형 데이터 >> softmax >> 객체 식별 가능!

model.summary()

""" model.summary() Flatten 이전
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50
 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205
 dense (Dense)               (None, 3, 3, 10)          60
=================================================================
Total params: 315
Trainable params: 315
Non-trainable params: 0 """
""" Flatten, Reshape : 데이터의 순서, 내용은 유지, 형태만 변경
Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50
 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205
 flatten (Flatten)           (None, 45)                0 
 dense (Dense)               (None, 10)                460
 dense_1 (Dense)             (None, 10)                110
=================================================================
Total params: 825
Trainable params: 825
Non-trainable params: 0 """
