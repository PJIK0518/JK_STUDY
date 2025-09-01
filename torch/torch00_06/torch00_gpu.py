import torch

# pytorch 버전확인
print('Pytorch 버전확인 :', torch.__version__)

# CUDA  사용여부 확인
CUDA_available = torch.cuda.is_available()
print('CUDA 사용여부    :', CUDA_available)

# 사용가능한 GPU 개수 확인
GPU_count = torch.cuda.device_count()
print('사용가능 GPU 개수:', GPU_count)

if CUDA_available:
    #현재 사용중인 GPU 장치 확인
    current_device = torch.cuda.current_device()
    print('현재 사용중인 GPU ID :', current_device)
    print('현재 GPU 이름        :', torch.cuda.get_device_name(current_device))
else:
    print('GPU 없음')
    
# CUDA 버전 확인
print('CUDA 버전 :', torch.version.cuda)

# CUDNN 버전 확인
CUDNN_version = torch.backends.cudnn.version()

if CUDNN_version is not None:
    print('CUDNN 버전:', CUDNN_version)
else :
    print('CUDNN 없음')
    
# Pytorch 버전확인 : 2.7.1+cu118
# CUDA 사용여부    : True
# 사용가능 GPU 개수: 1
# /home/jaeik/anaconda3/envs/JK_TORCH_271/lib/python3.9/site-packages/torch/cuda/__init__.py:287: UserWarning: 
# NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
# The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_37 sm_90.
# If you want to use the NVIDIA GeForce RTX 5070 Ti GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

#   warnings.warn(
# 현재 사용중인 GPU ID : 0
# 현재 GPU 이름        : NVIDIA GeForce RTX 5070 Ti
# CUDA 버전 : 11.8
# CUDNN 버전: 90100