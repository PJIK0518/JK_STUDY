import tensorflow as tf
# print(tf.__version__)   2.9.3

gpus = tf.config.list_physical_devices('GPU')
print(gpus)

if gpus:
    print('GPU 있다!!!')
else:
    print('GPU 없다...')
    
# GPU 있다!!!
# tensorflow의 경우에는 GPU와 CPU를 인식하는 버전이 따로 있다!