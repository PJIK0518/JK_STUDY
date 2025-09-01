#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
남/녀 얼굴 데이터(학습) + 내 얼굴(테스트)에 노이즈 추가 → Denoising Conv AutoEncoder 학습
학습된 가중치로 내 사진 복원 + 감마보정(미백 효과)
최종 결과: 내 원본 / 노이즈 / 복원 / 미백
"""

import os, random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ----------------------
# 0) 경로/설정
# ----------------------
PATH = './Study25/_data/kaggle/men_women/'
TRAIN_NPY = os.path.join(PATH, 'x_trn.npy')   # (N, 200, 200, 3) 가정
MY_IMG    = os.path.join(PATH, '박재익.jpg')   # 개인 사진
TEST_NPY  = os.path.join(PATH, 'x_tst.npy')   # 생성/캐시용

IMG_H, IMG_W, IMG_C = 200, 200, 3
SEED = 42
np.random.seed(SEED); random.seed(SEED)

# ----------------------
# 1) 데이터 로드 & 정규화
# ----------------------
x_train = np.load(TRAIN_NPY)  # (N, 200, 200, 3)
# 정규화: 0~1
if x_train.dtype != np.float32 and x_train.max() > 1.5:
    x_train = x_train.astype(np.float32) / 255.0
else:
    x_train = x_train.astype(np.float32)

# 내 사진 로드 → (1, H, W, 3)
img = load_img(MY_IMG, target_size=(IMG_H, IMG_W))
arr = img_to_array(img)  # (H, W, 3), float32 0~255
if arr.max() > 1.5:
    arr = arr / 255.0
x_test = arr[np.newaxis, ...]  # (1, H, W, 3)
np.save(TEST_NPY, x_test)

# ----------------------
# 2) 노이즈 추가 (가우시안)
# ----------------------
def add_gaussian_noise(x, sigma=0.07):
    noisy = x + np.random.normal(0, sigma, size=x.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)

x_train_noised = add_gaussian_noise(x_train, sigma=0.07)
x_test_noised  = add_gaussian_noise(x_test,  sigma=0.07)

# ----------------------
# 3) 모델: U-Net 스러운 간단한 DAE (3채널 입출력)
# ----------------------
def build_autoencoder(base_filters: int = 64):
    inp = Input(shape=(IMG_H, IMG_W, IMG_C))

    # Encoder
    c1 = Conv2D(base_filters, (3,3), activation='relu', padding='same')(inp)
    p1 = MaxPool2D((2,2), padding='same')(c1)

    c2 = Conv2D(32, (3,3), activation='relu', padding='same')(p1)
    p2 = MaxPool2D((2,2), padding='same')(c2)

    # Bottleneck
    b  = Conv2D(32, (3,3), activation='relu', padding='same')(p2)

    # Decoder + Skip
    u3 = UpSampling2D((2,2))(b)
    m3 = Concatenate(axis=-1)([u3, c2])
    c3 = Conv2D(16, (3,3), activation='relu', padding='same')(m3)

    u4 = UpSampling2D((2,2))(c3)
    m4 = Concatenate(axis=-1)([u4, c1])
    c4 = Conv2D(16, (3,3), activation='relu', padding='same')(m4)

    # 출력 3채널 + sigmoid (0~1)
    out = Conv2D(3, (3,3), activation='sigmoid', padding='same')(c4)
    return Model(inp, out)

model = build_autoencoder(64)
model.compile(optimizer=Adam(1e-3), loss='mae')  # mae가 bce/mse보다 덜 흐릿해지는 경우가 많음

# 검증 분리
tr_noisy, va_noisy, tr_clean, va_clean = train_test_split(
    x_train_noised, x_train, test_size=0.15, random_state=SEED, shuffle=True
)

# 학습
history = model.fit(
    tr_noisy, tr_clean,
    validation_data=(va_noisy, va_clean),
    epochs=20, batch_size=32, verbose=1
)

# ----------------------
# 4) 내 사진 복원 + 미백(감마 보정)
# ----------------------
deno_test = model.predict(x_test_noised, verbose=0)  # (1, H, W, 3)

def gamma_correct(x, gamma=0.85):
    # gamma < 1 → 밝게, >1 → 어둡게
    x = np.clip(x, 0.0, 1.0)
    return np.power(x, gamma)

whiten_test = gamma_correct(deno_test, gamma=0.85)

# ----------------------
# 5) 시각화: 내 사진 (원본 / 노이즈 / 복원 / 미백)
# ----------------------
show_list  = [x_test[0], x_test_noised[0], deno_test[0], whiten_test[0]]
show_title = ['원본', '노이즈', '복원', '미백(감마0.85)']

plt.figure(figsize=(4*len(show_list), 4))
for i, (img_, t_) in enumerate(zip(show_list, show_title), start=1):
    plt.subplot(1, len(show_list), i)
    plt.imshow(np.clip(img_, 0, 1))
    plt.title(t_, fontsize=12)
    plt.axis('off')
plt.tight_layout()
plt.show()

# ----------------------
# 6) 필요 시 결과 저장 (PNG)
# ----------------------
out_dir = os.path.join(PATH, "_out")
os.makedirs(out_dir, exist_ok=True)

def save_img01(x, fname):
    from PIL import Image
    arr = (np.clip(x,0,1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(out_dir, fname))

save_img01(x_test[0],       "00_original.png")
save_img01(x_test_noised[0],"01_noisy.png")
save_img01(deno_test[0],    "02_denoised.png")
save_img01(whiten_test[0],  "03_whitened.png")

print(f"[OK] 저장 완료 → {out_dir}")
