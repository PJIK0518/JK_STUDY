from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

Augment_size = 100                    # 증가시킬 사이즈

(x_trn, y_trn), (x_tst, y_tst) = fashion_mnist.load_data()

# print(x_trn.shape)      (60000, 28, 28)
# print(x_trn[0].shape)   (28, 28)
### x_trn[0] 하나만 100개로 증폭

# plt.imshow(x_trn[0], cmap='gray')
# plt.show()              # 신발 1개


aaa = np.tile(x_trn[0].reshape(28*28), Augment_size).reshape(-1,28,28,1)

       # np.tile(A, n) : A 데이터를 n번 만큼 복사
# (28, 2800) : reshape(28*28) X, reshape(-1,28,28,1) X
# (78400,)   : reshape(28*28) O, reshape(-1,28,28,1) X
# (100, 28, 28, 1)
# print(type(aaa)) <class 'numpy.ndarray'>
# print(aaa.shape) (100, 28, 28, 1)
#####################
### 데이터 증폭

IDG = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest' 
)

xy_data = IDG.flow(
    aaa,                         # x 데이터
    np.zeros(Augment_size),      # y 데이터 생성, Augment_size 만큼 0 생성
    batch_size=Augment_size,     # 통배치      
    shuffle=False,
).next()                         # xy_data를 Iterator 안의 모든 데이터를 꺼낸 상태로 정의

print(xy_data)                   # next가 있을 때 : 모든 데이터 // 없을 때 : NumpyArrayIterator라고 데이터 유형이 출력
print(type(xy_data))             # <class 'tuple'>
print(len(xy_data))              # next가 있을 때 : 2 // 없을 때 : 1
print(xy_data[0].shape)          # (100, 28, 28, 1)
print(xy_data[1].shape)          # (100,)

plt.figure(figsize=(7,7))

for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][i], cmap='gray')
    plt.axis('off')

plt.show()