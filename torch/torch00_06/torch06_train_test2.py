############################################################
#0. 준비
############################################################
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optm

from torch.utils.data import random_split, TensorDataset

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
x = np.array(range(100))
y = np.array(range(1,101))
x_prd = np.array([101, 102])

trn_size = int(0.8 * len(x))
tst_size = len(x) - trn_size

### [함수] 데이터 torch화
def TORCH(X):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    return X

x = TORCH(x)
y = TORCH(y)
x_prd = TORCH(x_prd)

g = torch.Generator().manual_seed(RS) 

DS = TensorDataset(x, y)

trn, tst = random_split(
    DS, [trn_size, tst_size],
    generator=g)

x_trn = torch.stack([x for x, y in trn]).to(DEVICE)
x_tst = torch.stack([x for x, y in tst]).to(DEVICE)
y_trn = torch.stack([y for x, y in trn]).to(DEVICE)
y_tst = torch.stack([y for x, y in tst]).to(DEVICE)

############################################################
#2. 모델
############################################################
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1)).to(DEVICE)

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
    trn_loss = TRAIN(model, loss, optimizer, x_trn, y_trn)
    print('epo :', e)
    print('mse :', trn_loss)

############################################################
#4. 예측 평가
############################################################
def EVALUATE(MODL, LOSS, XTST, YTST):
    MODL.eval()
    with torch.no_grad():
         x_tst_prd = MODL(XTST)
         tst_loss = LOSS(YTST, x_tst_prd)
    return tst_loss

tst_loss = EVALUATE(model, loss, x_tst, y_tst)
result = model(x_prd)

print('# ✅ 최종 성능 :', tst_loss.item())
print('# 🔍 예측 결과 :', '\n', f'{np.round(result.cpu().detach().numpy()[:])}')

# ✅ 최종 성능 : 0.0010027375537902117
# 🔍 예측 결과 : 
#  [[102.]
#   [103.]]