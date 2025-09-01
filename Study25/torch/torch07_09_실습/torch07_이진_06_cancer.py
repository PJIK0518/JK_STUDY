import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
DS = load_breast_cancer()
x = DS.data
y = DS.target

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
y_trn = torch.tensor(y_trn, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_tst = torch.tensor(y_tst, dtype=torch.float32).unsqueeze(1).to(DEVICE)

print('🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹')

############################################################
#2. 모델
############################################################
model = nn.Sequential(
    nn.Linear(30, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16,  1),
    nn.Sigmoid()).to(DEVICE)

EPOCHS = 1000

############################################################
#3. 컴파일 훈련
############################################################
loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def TRAIN(MODL, LOSS, OPTM, XTRN, YTRN):
    MODL.train()
    OPTM.zero_grad()
    x_trn_prd = MODL(XTRN)
    trn_loss = LOSS(x_trn_prd, YTRN)
    trn_loss.backward()
    OPTM.step() 
    return trn_loss.item()

for e in range(1, EPOCHS+1):
    trn_loss = TRAIN(model, loss, optimizer, x_trn, y_trn)
    print('epo :', e)           # verbose 를 직접 설정
    print('bce :', trn_loss)
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