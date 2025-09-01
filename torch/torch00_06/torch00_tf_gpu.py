import tensorflow as tf
print("텐서플로우 버전 :", tf.__version__)

if tf.config.list_physical_devices('GPU'):
    print("GPU 있음")
else:
    print("GPU 없음")

# CUDA 버전
cuda_version = tf.sysconfig.get_build_info()['cuda_version']
print("CUDA 버전 :", cuda_version)

# cuDNN 버전
cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
print("cuDNN 버전 :", cudnn_version)

# 텐서플로우 버전 : 2.15.0
# GPU 있음
# CUDA 버전 : 12.2
# cuDNN 버전 : 8