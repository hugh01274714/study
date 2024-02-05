from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
import time
from sklearn.datasets import load_boston

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 46)

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13)) 
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1) # patience=20 >> 20epoch까지 최소값을 기다리겠다.

hist = model.fit(x_train, y_train, epochs=10000, batch_size=1,
          validation_split=0.3, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print('r2 스코어:', r2)
 

print(hist.history['val_loss'])
'''
import matplotlib.pyplot as plt 
plt.figure(figsize=(9,5)) # 판을 깔다.
plt.plot(hist.history['loss'], marker= '.', c='red', label='loss') # 선, 점을 그리다.
plt.plot(hist.history['val_loss'], marker= '.', c='blue', label='val_loss') 
plt.grid() # 격자를 보이게
plt.title('loss') # 제목
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()
'''


'''
val_loss의 최하점을 찾는 것은 #3.훈련에서 이며, 
hist의 최저점을 찾으면 된다. 
EarlyStopping의 신뢰성은??? >>
patience는 최소에서 멈추는 것이 아닌 최소에서 50번 지난 지점에서 멈춘다.

#과제 
최소값에서 100번 지나서 멈춘거라면, 안좋은거기 때문에 
부수적인 파라미터가 있다면, 명시할 것 
디아벳까지 할 것 
hist값과 loss값과 


'''