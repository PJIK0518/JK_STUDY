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
x_trn = np.array([1,2,3,4,5,6,7])
y_trn = np.array([1,2,3,4,5,6,7])

x_tst = np.array([8,9,10,11])
y_tst = np.array([8,9,10,11])

x_prd = np.array([12, 13, 14])

def TORCH(X):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    return X

x_trn = TORCH(x_trn)
y_trn = TORCH(y_trn)
x_tst = TORCH(x_tst)
y_tst = TORCH(y_tst)
x_prd = TORCH(x_prd)

# print(x_trn.size())
# print(x_tst.size())
# print(y_trn.size())
# print(y_tst.size())
# print(x_prd.size())
# exit()
############################################################
#2. 데이터
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

# ✅ 최종 성능 : 3.676518645079341e-08
# 🔍 예측 결과 : 
#  [[12.]
#   [13.]
#   [14.]]