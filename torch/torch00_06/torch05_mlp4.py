############################################################
#0. 준비
############################################################
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optm

### 랜덤고정
RS = 55
torch.manual_seed(RS)
np.random.seed(RS)
random.seed(RS)

### GPU 설정
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA
                             else 'CPU')

############################################################
#1. 데이터
############################################################
x = np.array([range(10)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]]).T

x_prd = [[10]]

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
x_prd = torch.tensor(x_prd, dtype=torch.float32).to(DEVICE)

print(x.size())
print(y.size())
print(x_prd.size())

############################################################
#2. 모델
############################################################
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3)).to(DEVICE)

EPOCHS = 1000

############################################################
#3. 컴파일 훈련
############################################################
loss = nn.MSELoss()
optimizer = optm.Adam(model.parameters(), lr=0.002)

def TRAIN(MODL, LOSS, OPTM, XTRN, YTRN):
    MODL.train()
    OPTM.zero_grad()
    x_trn_prd = MODL(XTRN)
    trn_loss = LOSS(YTRN, x_trn_prd)
    trn_loss.backward()
    OPTM.step()
    return trn_loss

for e in range(1, EPOCHS+1):
    trn_loss = TRAIN(model, loss, optimizer, x, y)
    print(f'epo : {e}')
    print(f'mse : {trn_loss.item()}')
    
############################################################
#4. 평가 예측
############################################################
def EVALUATE(MODL, LOSS, XTST, YTST):
    MODL.eval()
    
    with torch.no_grad():
         x_tst_prd = MODL(XTST)
         tst_loss = LOSS(YTST, x_tst_prd)
    
    return tst_loss

tst_loss = EVALUATE(model, loss, x, y)

result = model(x_prd)

print('# ✅ 최종 성능 :', tst_loss.item())
print(f'# 🔍 예측 결과 : {np.round(result.cpu().detach().numpy()[0])}')

# ✅ 최종 성능 : 2.1583921715889742e-12
# 🔍 예측 결과 : [11.  0. -1.]