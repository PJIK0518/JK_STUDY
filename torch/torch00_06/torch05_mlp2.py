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
""" print(torch.__version__) 2.7.1+cu128 """
""" print(DEVICE)            cuda """

############################################################
#1. 데이터
############################################################
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.dtype)
# float64 : np의 기본 형태
#           > torch는 32를 처리해야함

x = np.transpose(x)

""" print(x.shape) (10, 3) """
""" print(y.shape) (10,) """

# x = torch.FloatTensor(x).to(DEVICE)
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

""" print(x.size()) torch.Size([10, 3]) """
""" print(y.size()) torch.Size([10, 1]) """

### Scaling
min = torch.min(x)
max = torch.max(x)

x = (x - min) / (max - min)

############################################################
#2. 모델
############################################################
model = nn.Sequential(
    nn.Linear(3,5),
    nn.ReLU(),
    nn.Linear(5,3),
    nn.Linear(3,1)).to(DEVICE)

EPOCHS = 1000

############################################################
#3. 컴파일 훈련
############################################################
loss = nn.MSELoss()
optimizer = optm.Adam(model.parameters(), lr=0.005)

def TRAIN(MODL, LOSS, OPTM, XTRN, YTRN):
    MODL.train()
    
    OPTM.zero_grad()
    
    x_trn_prd = MODL(XTRN)
    
    trn_loss = LOSS(YTRN, x_trn_prd)
    
    trn_loss.backward()
    
    OPTM.step()
    
    return trn_loss.item()

for e in range(1, EPOCHS+1):
    trn_loss = TRAIN(model, loss, optimizer, x, y)
    print(f'epo : {e:0d}')
    print(f'mse : {trn_loss}')
    
print('🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹')
############################################################
#4. 평가 예측
############################################################
def EVALUATE(MODL, LOSS, XTST, YTST):
    MODL.eval()
    
    with torch.no_grad():
         x_tst_prd = MODL(XTST)
         tst_loss = LOSS(x_tst_prd, YTST)
    
    return tst_loss.item()

x_prd = np.array([[11, 2.0, -1]])
x_prd = torch.FloatTensor(x_prd).to(DEVICE)
x_prd = (x_prd - min) / (max - min)

result = model(x_prd)
tst_loss = EVALUATE(model, loss, x, y)

print('# ✅ 최종 성능 :', tst_loss)
print('# 🔍 예측 결과 :', result.item())

# ✅ 최종 성능 : 0.00022938151960261166
# 🔍 예측 결과 : 11.003242492675781