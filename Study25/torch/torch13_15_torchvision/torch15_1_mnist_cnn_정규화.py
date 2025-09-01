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