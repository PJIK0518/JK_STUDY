##################################################
#0. 준비
##################################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd

from sklearn.datasets import load_iris
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
DS = load_iris()
x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=RS
)

   
x_trn = torch.tensor(x_trn, dtype=torch.float32).to(DEVICE)
x_tst = torch.tensor(x_tst, dtype=torch.float32).to(DEVICE)
y_trn = torch.tensor(y_trn, dtype=torch.long).to(DEVICE)
y_tst = torch.tensor(y_tst, dtype=torch.long).to(DEVICE)

# 다양한 플랫폼 (특히 GPU)에서 타입 충돌을 피하기 위해,
# PyTorch는 Target Label은 무조건 int64(long)으로 고정

""" print(x_trn.shape) torch.Size([120, 4]) """
""" print(y_trn.shape) torch.Size([120, 1]) """

##################################################
#2. 모델
##################################################
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.Linear(4, 3),
    ).to(DEVICE)

EPOCHS = 5000

##################################################
#3. 컴파일 훈련
##################################################
loss = nn.CrossEntropyLoss() # Sparse_categorical_crossentropy
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def TRAIN(MODL, LOSS, OPTM, XTRN, YTRN):
    MODL.train()
    OPTM.zero_grad()
    x_trn_prd = MODL(XTRN)
    trn_cce = LOSS(x_trn_prd, YTRN)
    trn_cce.backward()
    OPTM.step()
    return trn_cce.item()

for e in range(1, EPOCHS+1):
    trn_cce = TRAIN(model, loss, optimizer, x_trn, y_trn)
    print('epo :', f'{e}')
    print('cce :', f'{trn_cce}')

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
# x_tst_prd = np.argmax(x_tst_prd, axis=1)
y_tst = y_tst.cpu().detach().numpy()

acc = accuracy_score(x_tst_prd, y_tst)

print('loss :', tst_cce.item())
print('accu :', acc)

# loss : 0.060974862426519394
# accu : 0.9666666666666667