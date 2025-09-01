import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd

from sklearn.datasets import load_breast_cancer
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

############################################################
#1. 데이터
############################################################
path = './Study25/_data/kaggle/bank/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)

LE_GEO = LabelEncoder()
LE_GEN = LabelEncoder() 

trn_csv['Geography'] = LE_GEO.fit_transform(trn_csv['Geography'])
trn_csv['Gender'] = LE_GEN.fit_transform(trn_csv['Gender'])

trn_csv = trn_csv.drop(['CustomerId','Surname'], axis=1)

x = trn_csv.drop(['Exited'], axis=1)
y = trn_csv['Exited']


x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    random_state=RS
)

Scaler = StandardScaler()
x_trn = Scaler.fit_transform(x_trn)
x_tst = Scaler.transform(x_tst)

x_trn = torch.tensor(x_trn, dtype=torch.float32).to(DEVICE)
x_tst = torch.tensor(x_tst, dtype=torch.float32).to(DEVICE)
y_trn = torch.tensor(y_trn.to_numpy(), dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_tst = torch.tensor(y_tst.to_numpy(), dtype=torch.float32).unsqueeze(1).to(DEVICE)

### Dataloader
from torch.utils.data import TensorDataset # 데이터 합치기
from torch.utils.data import DataLoader    # Batch 나누기

trn_set = TensorDataset(x_trn, y_trn)
tst_set = TensorDataset(x_tst, y_tst)

trn_loader = DataLoader(trn_set, batch_size=100, shuffle=True)
tst_loader = DataLoader(tst_set, batch_size=100, shuffle=False)

print('🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹')

############################################################
#2. 모델
############################################################
# model = nn.Sequential(
#     nn.Linear(10, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.Linear(16,  1),
#     nn.Sigmoid()).to(DEVICE)

class Model(nn.Module):
    def __init__(self, ID, OD):
        super().__init__()
        self.lin1 = nn.Linear(ID, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 16)
        self.lin5 = nn.Linear(16, OD)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin4(x)
        x = self.lin5(x)
        x = self.sigm(x)
        
        return x

model = Model(10, 1).to(DEVICE)        

EPOCHS = 1000

############################################################
#3. 컴파일 훈련
############################################################
loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def TRAIN(MODL, LOSS, OPTM, trn_loader): # epo 단위의 훈련 > batch 단위의 훈련
    MODL.train()
    
    total_loss = 0
    
    for XTRN, YTRN in trn_loader:
        OPTM.zero_grad()
        
        x_trn_prd = MODL(XTRN)
        trn_loss = LOSS(x_trn_prd, YTRN)
        
        trn_loss.backward()
        OPTM.step() 
        total_loss = total_loss + trn_loss.item()
        total_loss += trn_loss.item()
        
    return total_loss / len(trn_loader)

for e in range(1, EPOCHS+1):
    trn_loss = TRAIN(model, loss, optimizer, trn_loader)
    print('epo :', e)
    print('mse :', trn_loss)


print('🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹')
############################################################
#4. 평가 예측
############################################################
def EVALUATE(MODL, LOSS, XTST, YTST):
    MODL.eval()
    
    with torch.no_grad():
         x_tst_prd = MODL(XTST)
         tst_loss = LOSS(x_tst_prd, YTST)
    
    return tst_loss.item()

tst_loss = EVALUATE(model, loss, x_tst, y_tst)

x_tst_prd = model(x_tst)

# print(type(x_tst_prd)) <class 'torch.Tensor'> 
# :: x_tst_prd 및 y_tst는 torch.Tensor 형태 
# :: accuracy_score는 numpy 형태로 cpu에서 계산 

y_tst = y_tst.detach().cpu().numpy()
x_tst_prd = np.round(x_tst_prd.detach().cpu().numpy())

acc = accuracy_score(y_tst, x_tst_prd)

print('bce :', tst_loss)
print('acc :', acc)

# bce : 0.7100207209587097
# acc : 0.9707602339181286