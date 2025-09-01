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
""" Default
                                                              Layer_Type Layer_Name  Layer_Able
0      <keras.src.engine.functional.Functional object at 0x768bd1b64df0>      vgg16        True
1  <keras.src.layers.reshaping.flatten.Flatten object at 0x768bd1b67880>    flatten        True
2           <keras.src.layers.core.dense.Dense object at 0x768bd1b67c10>      dense        True
3           <keras.src.layers.core.dense.Dense object at 0x768bd1b909d0>    dense_1        True """
##############################
#1. 전체 동결
""" model.trainable = False
                                                              Layer_Type Layer_Name  Layer_Able
0      <keras.src.engine.functional.Functional object at 0x7c90f6decee0>      vgg16       False
1  <keras.src.layers.reshaping.flatten.Flatten object at 0x7c90f6def9a0>    flatten       False
2           <keras.src.layers.core.dense.Dense object at 0x7c90f6deffd0>      dense       False
3           <keras.src.layers.core.dense.Dense object at 0x7c90f6e18af0>    dense_1       False """

#2. 전체 동결 반복문
for layer in model.layers:
    layer.trainable = False
"""                                                               Layer_Type Layer_Name  Layer_Able
0      <keras.src.engine.functional.Functional object at 0x7904d67f0f40>      vgg16       False
1  <keras.src.layers.reshaping.flatten.Flatten object at 0x7904d67f3a00>    flatten       False
2           <keras.src.layers.core.dense.Dense object at 0x7904d67f3d90>      dense       False
3           <keras.src.layers.core.dense.Dense object at 0x7904d681cb50>    dense_1       False """

#3. 부분동결 : 통상적으로 전이학습을 진행할 때 주로 사용
""" model.layers[0].trainable = False
                                                              Layer_Type Layer_Name  Layer_Able
0      <keras.src.engine.functional.Functional object at 0x793dd5d70e80>      vgg16       False
1  <keras.src.layers.reshaping.flatten.Flatten object at 0x793dd5d73940>    flatten        True
2           <keras.src.layers.core.dense.Dense object at 0x793dd5d73f70>      dense        True
3           <keras.src.layers.core.dense.Dense object at 0x793dd5d9ca90>    dense_1        True """

import pandas as pd
pd.set_option('max_colwidth', None)               # 터밀널에 출력되는 조건 설정
                                                  # None : 싹 다 나오게 / 정수형으로 몇 글자 출력할지 설정가능
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]

result = pd.DataFrame(layers, columns=['Layer_Type', 'Layer_Name','Layer_Able'])

print(result)