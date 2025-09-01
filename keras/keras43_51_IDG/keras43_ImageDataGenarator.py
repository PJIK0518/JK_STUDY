### 20250613
### brain

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

# 이미지 파일의 수치화
trn_IDG = ImageDataGenerator(
    rescale=1./255.,                # 0~255 로 정규화
    horizontal_flip=True,           # 수평 반전 : 데이터의 증폭 및 변환
    vertical_flip=True,             # 수직 반전 : 데이터의 증폭 및 변환
    width_shift_range=0.1,          # 평행이동 10%
    height_shift_range=0.1,
    rotation_range=5,               # 회전 5도
    zoom_range=1.2,                 # 확대 1.2배
    shear_range=0.7,                # 왜곡, 기준 좌표를 잡고 왜곡되는 정도
    fill_mode='nearest'             # 소실 데이터의 경우 주변 수치의 근사값으로 
)

tst_IDG = ImageDataGenerator(       # 테스트 데이터의 경우에 수정 없이 모델에 입력되는 데이터
    rescale=1./255.,                # 수치화만 진행
)

# 파일 데이터 경로 설정
path_trn = './_data/image/brain/train/'
path_tst = './_data/image/brain/test/'
                                    # 현재 train 폴더 안에 있는 폴더별로 인식 가능
                                    
xy_trn = trn_IDG.flow_from_directory(        # 특정 폴더에서 가져와서 인스턴스화된 기능을 먹여라
    path_trn,                                # 경로
    target_size=(200, 200),                  # Resize, 모든 데이터를 동일한 사이즈로 규격화
    batch_size=10,                           # 총 데이터 : (160,200,200,1) > 16*(10,200,200,1)
    class_mode='binary',                     # 이진분류 // 다중분류 : categorical
    color_mode='grayscale',                  
    shuffle=True,
    seed=123,                                # ImageDataGenerator_shuffle의 random_state
)                                            # Found 160 images belonging to 2 classes

xy_tst = tst_IDG.flow_from_directory(        
    path_tst,                                
    target_size=(200, 200),                 
    batch_size=10,          
    class_mode='binary',          
    color_mode='grayscale',
    # shuffle=True,                          # Test는 꼭 안해도 됨
)                                            # Found 120 images belonging to 2 classes

""" print(xy_trn.class_indices) {'ad': 0, 'normal': 1} """
""" print(xy_trn) <keras.preprocessing.image.DirectoryIterator object at 0x0000018CEEDE1100>
                    DirectoryIterator : 경로에서 반복되는 형태의 데이터를 순서대로 꺼내오는 기능
"""
""" print(xy_trn[0]) : brain의 ad : 
    X : 너무 길어서 생략... 10개의 수치화된 이미지 데이터
    y : rray[1., 0., 0., 0., 1., 0., 0., 1., 0., 1.]
"""
""" print(xy_trn[0][1]) [1. 1. 1. 1. 0. 1. 1. 1. 0. 0.] """
""" print(len(xy_trn)) 16 : (총 데이터)/(batch)"""
""" print(xy_trn[0][0].shape) (10, 200, 200, 1) """
""" print(xy_trn[0][1].shape) (10,) """
""" print(xy_trn[0][1]) : batch = 160 > x_trn, y_trn으로 분할에 용이
[1. 1. 1. 1. 0. 1. 1. 1. 0. 0. 1. 0. 0. 1. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 1. 0. 0. 1. 1. 1. 1. 1. 0. 1. 0. 0. 0. 0. 1. 0. 1. 1. 1. 0. 1.
 1. 1. 0. 0. 0. 1. 1. 1. 1. 0. 1. 0. 1. 0. 1. 1. 1. 1. 0. 1. 1. 0. 0. 0.
 0. 0. 1. 1. 1. 0. 0. 0. 1. 1. 1. 0. 1. 0. 1. 0. 0. 1. 1. 0. 0. 0. 0. 1.
 0. 0. 1. 0. 0. 1. 0. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 1.
 0. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 0. 1. 0. 0. 0. 0. 1. 0. 1.
 1. 0. 0. 0. 1. 1. 1. 0. 1. 1. 1. 0. 1. 0. 1. 1.] """
""" print(type(xy_trn)) <class 'keras.preprocessing.image.DirectoryIterator'>
    : IDG로 변환시킨 데이터의 형태는 DirectoryIterator"""
""" print(type(xy_trn[0])) <class 'tuple'> 
    : xy_trn의 첫 번째 batch의 형태"""
""" print(type(xy_trn[0][0])) <class 'numpy.ndarray'>
    : xy_trn의 첫 번째 batch의 첫 째 컬럼(x데이터) 형태"""
""" print(type(xy_trn[0][1])) <class 'numpy.ndarray'>
    : xy_trn의 첫 번째 batch의 둘 째 컬럼(x데이터) 형태"""

### ERROR
""" print(xy_trn.shape) : AttributeError: 'DirectoryIterator' object has no attribute 'shape'
    : DirectoryIterator의 형태의 데이터는 len으로 몇 개짜리인지 확인 가능 """
""" print(xy_trn[0].shape) AttributeError: 'tuple' object has no attribute 'shape'
    : tuple 형태에 shape는 호환 불가"""
""" print(xy_trn[16]) ValueError: Asked to retrieve element 16, but the Sequence has length 16
    : 배치 수가 16개라서 0 ~ 15번 배치까지 있어서 발생하는 오류"""
""" print(xy_trn[0][2]) IndexError: tuple index out of range
    : 배치 내에 있는 데이터 유형이 두 개 뿐이어서 발생하는 오류"""



