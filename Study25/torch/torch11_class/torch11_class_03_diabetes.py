############################################################
#0. 준비
############################################################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

### GPU
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'CPU')

### 랜덤고정
RS = 55
torch.manual_seed(RS)
np.random.seed(RS)
random.seed(RS)

############################################################
#1. 데이터
############################################################
DS = load_diabetes()

x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    random_state=RS
)

### Scaling
SDS = StandardScaler()
SDS.fit(x_trn)
x_trn = SDS.transform(x_trn)
x_tst = SDS.transform(x_tst)

x_trn = torch.tensor(x_trn, dtype=torch.float32).to(DEVICE)
x_tst = torch.tensor(x_tst, dtype=torch.float32).to(DEVICE)
y_trn = torch.tensor(y_trn, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_tst = torch.tensor(y_tst, dtype=torch.float32).unsqueeze(1).to(DEVICE)

############################################################
#2. 모델
############################################################
# model = nn.Sequential(
#     nn.Linear(10, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16,  8),
#     nn.ReLU(),
#     nn.Linear( 8,  4),
#     nn.Linear( 4,  1)).to(DEVICE)

EPOCHS = 1000

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16,  8)
        self.linear4 = nn.Linear( 8,  4)
        self.linear5 = nn.Linear(4, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        
        x = self.linear2(x)
        x = self.relu(x)
        
        x = self.linear3(x)
        x = self.relu(x)
        
        x = self.linear4(x)
        
        x = self.linear5(x)
        
        return x
    
model = Model(10, 1).to(DEVICE)

############################################################
#3. 컴파일 훈련
############################################################
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def TRAIN(MODL, LOSS, OPTM, XTRN, YTRN):
    MODL.train()
    OPTM.zero_grad()
    x_trn_prd = MODL(XTRN)
    trn_loss = LOSS(x_trn_prd, YTRN)
    trn_loss.backward()
    OPTM.step()
    return trn_loss

for e in range(1, EPOCHS+1):
    trn_loss = TRAIN(model, loss, optimizer, x_trn, y_trn)
    print('epo :', e)
    print('mse :', trn_loss.item())

############################################################
#4. 평가 예측
############################################################
def EVALUATE(MODL, LOSS, XTST, YTST):
    MODL.eval()
    
    with torch.no_grad():
         x_tst_prd = MODL(XTST)
         tst_loss = LOSS(x_tst_prd, YTST)
    
    return tst_loss

tst_loss = EVALUATE(model, loss, x_tst, y_tst)
x_tst_prd = model(x_tst).cpu().detach().numpy()
y_tst = y_tst.cpu().detach().numpy()
r2 = r2_score(x_tst_prd, y_tst)

print(tst_loss.item())
print(r2)
# 3487.78125
# -1.280017375946045