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

# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda' if USE_CUDA else 'CPU')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'CPU')
print('torch :', torch.__version__)
print('dvice :', DEVICE)

##################################################
#1. 데이터
##################################################
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])

y = np.array([4,5,6,7,8,9,10])

# print(x.shape, y.shape)  (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
# print(x.shape)           (7, 3, 1)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

# print(x.shape, y.size()) torch.Size([7, 3, 1]) torch.Size([7])

from torch.utils.data import TensorDataset, DataLoader
trn_set = TensorDataset(x, y,)
trn_ldr = DataLoader(trn_set, batch_size=2, shuffle=True)

aaa = iter(trn_ldr)
bbb = next(aaa)
print(bbb)

##################################################
#2. 모델
##################################################
class RNN(nn.Module):
    def __init__(self):
        super(). __init__()
        self.RNN1 = nn.RNN(
            input_size = 1,    
            hidden_size = 32,
            num_layers = 1,
            batch_first = True,
            bidirectional = True)    # (Batch, TimeStep, features*2) 로 결과값 출력
                                     # reshape, 다음층의 input layer 등을 맞춰면 돌아는가지만
                                     # 결과값의 차원도 늘어나버림
        
        self.fc1 = nn.Linear(3*2*32, 16)
        # self.fc1 = nn.Linear(2*32, 16)  
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        
        self.rlu = nn.ReLU()
    
    def forward(self, x):
        x, _ = self.RNN1(x) 
        x = self.rlu(x)
        
        x = x.reshape(-1, 3*2*32)
        # x = x[:,-1,:]
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        x = self.rlu(x)
        
        x = self.fc3(x)
        
        return x

model = RNN().to(DEVICE)
from torchsummary import summary
""" summary(model, (3,1)) : x = x.reshape(-1, 32*3)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
               RNN-1  [[-1, 3, 32], [-1, 2, 32]]               0
              ReLU-2                [-1, 3, 32]               0
            Linear-3                   [-1, 16]           1,552
            Linear-4                    [-1, 8]             136
              ReLU-5                    [-1, 8]               0
            Linear-6                    [-1, 1]               9
================================================================
Total params: 1,697
Trainable params: 1,697
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.05
Params size (MB): 0.01
Estimated Total Size (MB): 0.05
---------------------------------------------------------------- """

""" summary(model, (3,1)) : x = x[:, -1, :] / self.fc1 = nn.Linear(32, 16)
        Layer (type)               Output Shape         Param #
================================================================
               RNN-1  [[-1, 3, 32], [-1, 2, 32]]               0
              ReLU-2                [-1, 3, 32]               0
            Linear-3                   [-1, 16]             528
            Linear-4                    [-1, 8]             136
              ReLU-5                    [-1, 8]               0
            Linear-6                    [-1, 1]               9
================================================================
Total params: 673
Trainable params: 673
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.05
Params size (MB): 0.00
Estimated Total Size (MB): 0.05
---------------------------------------------------------------- """

# RNN : x, h0
# 

# exit()

EPOCHS = 1000
##################################################
#3. 컴파일 훈련
##################################################
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.02)

def TRAIN(MODL, LOSS, OPTM, LODR):
    MODL.train()
    epo_lss = 0
    
    for XTRN, YTRN in LODR:
        XTRN, YTRN = XTRN.to(DEVICE), \
                     YTRN.to(DEVICE)
        
        OPTM.zero_grad()
        
        x_trn_prd = MODL(XTRN)
        trn_loss = LOSS(x_trn_prd, YTRN)
        
        trn_loss.backward()     # 기울기 계산
        OPTM.step()             # w = w - lr * 기울기
                
        epo_lss += trn_loss.item()
    
    return epo_lss / len(LODR)

def EVALUATE(MODL, LOSS, LODR):
    MODL.eval()
    total_loss = 0
    
    for XVAL, YVAL in LODR:
        XVAL, YVAL = XVAL.to(DEVICE), YVAL.to(DEVICE)
        with torch.no_grad():
            YPRD = MODL(XVAL)
            loss = LOSS(YPRD, YVAL)
            total_loss += loss.item()
                
    return total_loss / len(LODR)

for e in range(1, EPOCHS+1):
    trn_loss = TRAIN(model, loss, optimizer, trn_ldr)
    val_loss = EVALUATE(model, loss, trn_ldr)
    print(f'epo : {e}')
    print(f'trn_lss : {trn_loss:.5f}')
    print(f'val_lss : {val_loss:.5f}')

############################################################
#4. 평가 예측
############################################################
tst_loss = EVALUATE(model, loss, trn_ldr)

print(f'tst_lss : {tst_loss:.5f}')

x_prd = np.array([[8,9,10]])
x_prd = x_prd.reshape(x_prd.shape[0], x_prd.shape[1], 1)

x_prd  = torch.tensor(x_prd, dtype=torch.float32).to(DEVICE)

print(model(x_prd))

# bce : 0.7100207209587097
# acc : 0.9707602339181286

# z_scorenormalization
# tst_lss : 0.03532
# tst_acc : 0.9884