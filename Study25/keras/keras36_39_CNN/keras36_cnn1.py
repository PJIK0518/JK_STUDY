from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D
                                           # 2D 데이터에 대한 Convolution layer 제작 class

# 수업 : (N,5,5,1)의 이미지를 2*2로 convoltion
# 이미지는 기본적으로 4차원 데이터, 개수, 가로, 세로, 색

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1))) # 출력 (N,4,4,10)
                                                  # 10개의 channel로 출력,
                                                  # (2,2)의 kernel_size로,
                                                  # (n,5,5,1)의 데이터를 계산
model.add(Conv2D(5, (2,2))) # 출력 (N,3,3,5)
        # params은 연산량이 아니라 weight와 bias의 개수, 즉 수식의 개수! 계산량이 아니라 계산 식의 양!
        # Conv2D(filter, (kernel_size), (input_shape))

""" model.summary()
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50
 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205
=================================================================
Total params: 255
Trainable params: 255
Non-trainable params: 0
"""
""" model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,3)))

model.summary()
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_1 (Conv2D)           (None, 4, 4, 10)          130       
=================================================================
Total params: 130
Trainable params: 130
Non-trainable params: 0 """
