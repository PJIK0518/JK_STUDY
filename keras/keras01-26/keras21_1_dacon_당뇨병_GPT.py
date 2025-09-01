from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 로딩
path = './_data/dacon/당뇨병/'

train_df = pd.read_csv(path + "train.csv", index_col=0)
test_df = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sample_submission.csv", index_col=0)

# 2. 결측 처리: 0이 말이 안 되는 컬럼만 nan 처리
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
train_df[cols_with_zero] = train_df[cols_with_zero].replace(0, np.nan)
test_df[cols_with_zero] = test_df[cols_with_zero].replace(0, np.nan)

train_df[cols_with_zero] = train_df[cols_with_zero].fillna(train_df[cols_with_zero].mean())
test_df[cols_with_zero] = test_df[cols_with_zero].fillna(test_df[cols_with_zero].mean())

# 3. 피처/라벨 분리 + 스케일링
x = train_df.drop(['Outcome'], axis=1)
y = train_df['Outcome']

scaler = StandardScaler()
x = scaler.fit_transform(x)
test_df = scaler.transform(test_df)

# 4. 데이터 분할
x_trn, x_val, y_trn, y_val = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=55)

# 5. 모델 구성
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

model = build_model()

# 6. 컴파일 및 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)

history = model.fit(x_trn, y_trn,
                    epochs=1000,
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    callbacks=[es],
                    verbose=3)

# 7. 시각화
plt.figure(figsize=(9,6))
plt.title('Dacon 당뇨병')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history.history['loss'], color='red', label='train_loss')
plt.plot(history.history['val_loss'], color='green', label='val_loss')
plt.legend()
plt.grid()
plt.show()

# 8. 평가
y_pred_proba = model.predict(x_val)
y_pred = (y_pred_proba > 0.45).astype(int)

print("정확도 :", accuracy_score(y_val, y_pred))
print("AUC :", roc_auc_score(y_val, y_pred_proba))
print(classification_report(y_val, y_pred))
plt.show()

# 9. 제출 파일 생성
test_pred = model.predict(test_df)
test_pred = np.where(test_pred > 0.45, 1, 0)

submission['Outcome'] = test_pred
submission.to_csv(path + 'submission_optimized_0527_10.csv')