# Autoencoder를 왜 Dense로?? >>  Convolution AutoEncoder

# aa04_ae2_그림.py - capy
# 그림 그리는 plt 쪽만 바꿔보자 학습하며 이미지 변환한 순서대로 한꺼본에 뽑아 낼 거다 

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = (x_train.astype('float32') / 255.0)[..., np.newaxis]   # (60000, 28, 28, 1)
x_test  = (x_test.astype('float32')  / 255.0)[..., np.newaxis]   # (10000, 28, 28, 1)

x_train_noised = np.clip(x_train + np.random.normal(0, 0.1, size=x_train.shape), 0, 1)
x_test_noised  = np.clip(x_test  + np.random.normal(0, 0.1, size=x_test.shape ), 0, 1)

# 2. 모델
input_img = Input(shape=(28,28,1))

####[실습] : 딥하게 구성####
# Encoder (28, 28,  1)
# Conv2D  (28, 28, 64) padding = same
# maxpool (14, 14, 64)
# Conv2D  (14, 14, 32) padding = same
# maxpool ( 7,  7, 32)

# Decoder
# Conv2D  ( 7,  7, 32) padding = same
# UpSampling2D ( 2, 2) (14, 14, 32)
# Conv2D  (14, 14, 16) padding = same
# UpSampling2D ( 2, 2) (28, 28, 16)
# Conv2D  (28, 28,  1)

def autoencoder(f: int = 64):
    conv2d_1 = Conv2D(filters=f,
                    kernel_size=(2,2),
                    activation='relu',
                    padding= 'same')(input_img)
    maxp2d_1 = MaxPool2D(2,2,
                         padding= 'same')(conv2d_1)
    
    conv2d_2 = Conv2D(filters=32,
                    kernel_size=(2,2),
                    activation='relu',
                    padding= 'same')(maxp2d_1)
    maxp2d_2 = MaxPool2D(2,2,
                         padding= 'same')(conv2d_2)
    
    conv2d_3 = Conv2D(filters=32,
                    kernel_size=(2,2),
                    activation='relu',
                    padding= 'same')(maxp2d_2)
    upsamp_3 = UpSampling2D(size=(2,2))(conv2d_3)
    
    conv2d_4 = Conv2D(filters=16,
                    kernel_size=(2,2),
                    activation='relu',
                    padding= 'same')(upsamp_3)
    upsamp_4 = UpSampling2D(size=(2,2))(conv2d_4)
    
    conv2d_5 = Conv2D(filters=1,
                    kernel_size=(2,2),
                    activation='sigmoid',
                    padding= 'same')(upsamp_4)
    
    model = Model(input_img, conv2d_5)
    
    return model

model = autoencoder(64)

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)

decoded_imgs = model.predict(x_test_noised)

rlt_list = np.array(decoded_imgs)

import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import numpy as np

# (N,28,28)로 압축
orig   = x_test.squeeze(-1)
noised = x_test_noised.squeeze(-1)
deno   = decoded_imgs.squeeze(-1)

results = [orig, noised, deno]
titles  = ['원본', '노이즈', '복원']

num_rows, num_cols = len(results), 5
idxs = random.sample(range(orig.shape[0]), num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 3*num_rows))
for r in range(num_rows):
    for c, i in enumerate(idxs):
        axes[r, c].imshow(results[r][i], cmap='gray')
        if c == 0:
            axes[r, c].set_title(titles[r])
        axes[r, c].axis('off')

plt.tight_layout()
plt.show()