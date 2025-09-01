##################################################
#0. 준비
##################################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

### 랜덤고정
RS = 55
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
DS = fetch_covtype()
x = DS.data
y = DS.target

y = y-1

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=RS
)

x_trn = torch.tensor(x_trn, dtype=torch.float32).to(DEVICE)
x_tst = torch.tensor(x_tst, dtype=torch.float32).to(DEVICE)
y_trn = torch.tensor(y_trn, dtype=torch.int64).to(DEVICE)
y_tst = torch.tensor(y_tst, dtype=torch.int64).to(DEVICE)

""" print(x_trn.size()) torch.Size([464809, 54]) """
""" print(y_trn.size()) torch.Size([464809]) """
 
# print(torch.unique(y_trn, return_counts = True)[0])
# print(torch.unique(y_trn, return_counts = True)[1])
# tensor([1, 2, 3, 4, 5, 6, 7], device='cuda:0')
# tensor([169533, 226577,  28587,   2191,   7610,  13939,  16372],
#        device='cuda:0')

##################################################
#2. 모델
##################################################
# model = nn.Sequential(
#     nn.Linear(54, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64,  32),
#     nn.ReLU(),
#     nn.Linear(32,  16),
#     nn.Linear(16,  7),
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, ID, OD):
        super().__init__()
        self.lin1 = nn.Linear(ID, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 16)
        self.lin5 = nn.Linear(16, OD)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin4(x)
        x = self.lin5(x)
        
        return x

model = Model(54, 7).to(DEVICE)     

EPOCHS = 1000

##################################################
#3. 컴파일 훈련
##################################################
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

def TRAIN(MODL, LOSS, OPTM, XTRN, YTRN):
    MODL.train()
    OPTM.zero_grad()
    x_trn_prd = MODL(XTRN)
    trn_cce = LOSS(x_trn_prd, YTRN)
    trn_cce.backward()
    OPTM.step()
    return trn_cce

for e in range(1, EPOCHS+1):
    trn_cce = TRAIN(model, loss, optimizer, x_trn, y_trn)
    print('epo :', e)
    print('cce :', trn_cce.item())

##################################################
#4. 평가 예측
##################################################
def EVALUATE(MODL, LOSS, XTST, YTST):
    MODL.eval()
    
    with torch.no_grad():
         x_tst_prd = MODL(XTST)
         tst_cce = LOSS(x_tst_prd, YTST)
    
    return tst_cce

tst_cce = EVALUATE(model, loss, x_tst, y_tst)

x_tst_prd = model(x_tst)

x_tst_prd = torch.argmax(x_tst_prd, dim=1)
x_tst_prd = x_tst_prd.cpu().detach().numpy()
y_tst = y_tst.cpu().detach().numpy()

acc = accuracy_score(x_tst_prd, y_tst)

print('loss :', tst_cce.item())
print('accu :', acc)

# loss : 0.7115699648857117
# accu : 0.7062812491932222