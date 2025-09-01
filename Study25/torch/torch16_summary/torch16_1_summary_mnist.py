############### [실습] DROPOUT 적용 ###############
##################################################
#0. 준비
##################################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

### 랜덤고정
RS = 55
torch.cuda.manual_seed(RS)
torch.manual_seed(RS)
np.random.seed(RS)
random.seed(RS)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'CPU')
print('torch :', torch.__version__)
print('dvice :', DEVICE)

##################################################
#1. 데이터
##################################################
from torchvision.datasets import MNIST
import torchvision.transforms as tr
trsf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))])
# tr.Comppose : 여러 기능을 한번에 Pipeline화     
# tr.Resize   : 이미지 크기 규격화
# tr.ToTensro : torchtensor + Minamxscaler

########## 경우에 따라서 -1 ~ 1 사이로 적용하는게 더 좋을수도 있다..! ##########
# tr.Normalize : 표준화는 원래 (x - 평균) / 표편
# But. tr.Normalize((0.5),(0.5)) : (x - 0.5) / 0.5
#                                  -1~1 사이로 되고 계산도 편하고 성능도 적당
# = Z_score Normalization : 정규화의 표준화, 범위도 줄이고 평균 표편도 0, 1로...!
#############################################################################

#####  torch 이미지 : 컬러 가로 세로
##### tensro 이미지 : 가로 세로 컬러 >>> 그냥 reshape하면 안됨!!!!!

path = './Study25/_data/torch/'

trn_dataset = MNIST(path, train = True, download= True, transform= trsf)
tst_dataset = MNIST(path, train = False, download= True, transform= trsf)
# print(len(trn_dataset)) 60000
# print(trn_dataset[0][1]) 5

img_tensor, label = trn_dataset[0]

# print(img_tensor.shape) torch.Size([1, 56, 56]) : img_tensor = trn_dataset[0][0]
# print(label) 5                                  : label = trn_dataset[0][1]
# print(img_tensor.min(), img_tensor.max()) : tensor(0.) tensor(0.9922)
#                                           : MinMax까지 적용된 상태!!!!!   

### Dataloader
from torch.utils.data import TensorDataset # 데이터 합치기
from torch.utils.data import DataLoader    # Batch 나누기

trn_Loader = DataLoader(trn_dataset, batch_size=32, shuffle=True)
tst_Loader = DataLoader(tst_dataset, batch_size=32, shuffle=False)

print(len(trn_Loader)) # 1875 : trn_dataset / batch_size
print(len(tst_Loader)) # 313  : trn_dataset / batch_size
# exit()

##################################################
#2. 모델
##################################################
class CNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
    # = super(self, DNN).__init__()
        # input = 1* 56 *56
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),  # = 64* 54 *54
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2), # = 64* 27 *27
            nn.Dropout(0.2))
    
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1), # = 32* 25 *25
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), # = 32* 12 *12
            nn.Dropout(0.2))
    
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1), # = 16* 10 *10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), # = 16* 5 *5
            nn.Dropout(0.2))
        
        self.flatten = nn.Flatten()
    
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(400, 64),     # Linear 에서 Flatten
            nn.ReLU(),
            nn.Dropout(0.2))
            
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2))
               
        self.output_layer = nn.Linear(32, 10)
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.flatten(x)          # self.flatten을 안만들고
                                     # x = x.view(x.shape[0], -1)로 만들어도 가능
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        
        return x

model = CNN(1).to(DEVICE) # torch에서 Conv2D는 channel 값만 input,
                          # 뒤에 가로세로는 알아서 연산
EPOCHS = 100                

# model.summary() [ERROR] AttributeError: 'CNN' object has no attribute 'summary'

""" print(model) : __init__에 나오는 모델 : 우리가 원하는 부분이 아니다
CNN(
  (hidden_layer1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.2, inplace=False)
  )
  (hidden_layer2): Sequential(
    (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.2, inplace=False)
  )
  (hidden_layer3): Sequential(
    (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.2, inplace=False)
  )
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (hidden_layer4): Sequential(
    (0): Linear(in_features=400, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
  )
  (hidden_layer5): Sequential(
    (0): Linear(in_features=64, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
  )
  (output_layer): Linear(in_features=32, out_features=10, bias=True)
  (drop): Dropout(p=0.2, inplace=False)) """
# pip install torchsummary==1.5.1
from torchsummary import summary

""" summary(model, (1,56,56))   # channel, width, height
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 54, 54]             640
              ReLU-2           [-1, 64, 54, 54]               0
         MaxPool2d-3           [-1, 64, 27, 27]               0
           Dropout-4           [-1, 64, 27, 27]               0
            Conv2d-5           [-1, 32, 25, 25]          18,464
              ReLU-6           [-1, 32, 25, 25]               0
         MaxPool2d-7           [-1, 32, 12, 12]               0
           Dropout-8           [-1, 32, 12, 12]               0
            Conv2d-9           [-1, 16, 10, 10]           4,624
             ReLU-10           [-1, 16, 10, 10]               0
        MaxPool2d-11             [-1, 16, 5, 5]               0
          Dropout-12             [-1, 16, 5, 5]               0
          Flatten-13                  [-1, 400]               0
           Linear-14                   [-1, 64]          25,664
             ReLU-15                   [-1, 64]               0
          Dropout-16                   [-1, 64]               0
           Linear-17                   [-1, 32]           2,080
             ReLU-18                   [-1, 32]               0
          Dropout-19                   [-1, 32]               0
           Linear-20                   [-1, 10]             330
================================================================
Total params: 51,802
Trainable params: 51,802
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.97
Params size (MB): 0.20
Estimated Total Size (MB): 4.18
---------------------------------------------------------------- """

from torchinfo import summary
summary(model, (32,1,56,56))
""" summary(model, (32,1,56,56))   # batch, channel, width, height
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CNN                                      [32, 10]                  --
├─Sequential: 1-1                        [32, 64, 27, 27]          --
│    └─Conv2d: 2-1                       [32, 64, 54, 54]          640
│    └─ReLU: 2-2                         [32, 64, 54, 54]          --
│    └─MaxPool2d: 2-3                    [32, 64, 27, 27]          --
│    └─Dropout: 2-4                      [32, 64, 27, 27]          --
├─Sequential: 1-2                        [32, 32, 12, 12]          --
│    └─Conv2d: 2-5                       [32, 32, 25, 25]          18,464
│    └─ReLU: 2-6                         [32, 32, 25, 25]          --
│    └─MaxPool2d: 2-7                    [32, 32, 12, 12]          --
│    └─Dropout: 2-8                      [32, 32, 12, 12]          --
├─Sequential: 1-3                        [32, 16, 5, 5]            --
│    └─Conv2d: 2-9                       [32, 16, 10, 10]          4,624
│    └─ReLU: 2-10                        [32, 16, 10, 10]          --
│    └─MaxPool2d: 2-11                   [32, 16, 5, 5]            --
│    └─Dropout: 2-12                     [32, 16, 5, 5]            --
├─Flatten: 1-4                           [32, 400]                 --
├─Sequential: 1-5                        [32, 64]                  --
│    └─Linear: 2-13                      [32, 64]                  25,664
│    └─ReLU: 2-14                        [32, 64]                  --
│    └─Dropout: 2-15                     [32, 64]                  --
├─Sequential: 1-6                        [32, 32]                  --
│    └─Linear: 2-16                      [32, 32]                  2,080
│    └─ReLU: 2-17                        [32, 32]                  --
│    └─Dropout: 2-18                     [32, 32]                  --
├─Linear: 1-7                            [32, 10]                  330
==========================================================================================
Total params: 51,802
Trainable params: 51,802
Non-trainable params: 0
Total mult-adds (M): 444.69
==========================================================================================
Input size (MB): 0.40
Forward/backward pass size (MB): 53.33
Params size (MB): 0.21
Estimated Total Size (MB): 53.94
========================================================================================== """
exit()










##################################################
#3. 컴파일 훈련
##################################################
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1e-4) # 0.0001

def TRAIN(MODL, LOSS, OPTM, LODR):
    MODL.train()
    epo_lss = 0
    epo_acc = 0
    
    for XTRN, YTRN in LODR:
        XTRN, YTRN = XTRN.to(DEVICE), \
                     YTRN.to(DEVICE)
        
        OPTM.zero_grad()
        
        x_trn_prd = MODL(XTRN)
        trn_loss = LOSS(x_trn_prd, YTRN)
        
        trn_loss.backward()     # 기울기 계산
        OPTM.step()             # w = w - lr * 기울기
        
        YPRD = torch.argmax(x_trn_prd, 1)
        acc = (YPRD == YTRN).float().mean()
        
        epo_lss += trn_loss.item()
        epo_acc += acc
    
    return epo_lss / len(LODR), epo_acc / len(LODR)

def EVALUATE(MODL, LOSS, LODR):
    MODL.eval()
    total_loss = 0
    total_acc = 0
    
    for XVAL, YVAL in LODR:
        XVAL, YVAL = XVAL.to(DEVICE), YVAL.to(DEVICE)
        with torch.no_grad():
            YPRD = MODL(XVAL)
            loss = LOSS(YPRD, YVAL)
            total_loss += loss.item()
            
            YPRD_label = torch.argmax(YPRD, dim=1)
            acc = (YPRD_label == YVAL).float().mean()
            total_acc += acc.item()
    
    return total_loss / len(LODR), total_acc / len(LODR)

for e in range(1, EPOCHS+1):
    trn_loss, trn_acc = TRAIN(model, loss, optimizer, trn_Loader)
    val_loss, val_acc = EVALUATE(model, loss, tst_Loader)
    print(f'epo : {e}')
    print(f'trn_lss : {trn_loss:.5f} | trn_acc : {trn_acc:.5f}')
    print(f'val_lss : {val_loss:.5f} | val_acc : {val_acc:.5f}')

############################################################
#4. 평가 예측
############################################################
tst_loss, tst_acc = EVALUATE(model, loss, tst_Loader)

print(f'tst_lss : {tst_loss:.5f}')
print(f'tst_acc : {tst_acc:.5f}')

# bce : 0.7100207209587097
# acc : 0.9707602339181286

# z_scorenormalization
# tst_lss : 0.03532
# tst_acc : 0.9884