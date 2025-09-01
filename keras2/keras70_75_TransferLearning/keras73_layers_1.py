from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D


### VGG16 : CNN 16층 쌓아놓고 이미지넷이라는 대회 준우승한 모델
from tensorflow.keras.applications import VGG19, ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet152, ResNet152V2, DenseNet121, DenseNet169
from tensorflow.keras.applications import DenseNet201, InceptionV3, InceptionResNetV2

### 랜덤고정
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import numpy as np
import random
RS = 111
random.seed(RS)
np.random.seed(RS)
tf.random.set_seed(RS)


from tensorflow.keras.applications import VGG16

vgg16 = VGG16(include_top = False,
              input_shape = (32, 32, 3))

# vgg16.trainable = False

model = Sequential()

model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation = 'softmax'))

print(len(model.weights))          
print(len(model.trainable_weights))
# trainable = True  30 30
# trainable = False 30 4
### 실습
# [FLATTEN]
# 1. 전이학습 vs 기존최고기록 (acc  : 0.7702)
# 2. vgg 16 가중치 동결

# 3. vgg 16 가중치 안동결

# [GAP]
# 1. 전이학습 vs 기존최고기록 (acc  : 0.7702)
# 2. vgg 16 가중치 동결

# 3. vgg 16 가중치 안동결
