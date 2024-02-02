from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
import time
from sklearn.datasets import load_diabetes

#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 46)

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10)) 
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=5,
          validation_split=0.3)
# end = time.time() - start


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

'''
print('===================================')
print(hist)
print('===================================')
print(hist.history) # 딕셔너리 형태로 추출 됨 
print('===================================')
print(hist.history['loss'])
print('===================================')
print(hist.history['val_loss'])
print('===================================')
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
loss와 val_loss가 같이 내려가고 있다면, 어느 정도 훈련이 잘 되어가고 있다고 판단 가능하다.
val_loss가 최저점이다. 선과 데이터의 거리가 가장 가깝다 >> 가장 예측을 잘했다.
val_loss가 최저점인 지점을 저장해 놓아야 한다. 
hist를 사용해서 훈련 중 val_loss의 최저점에 멈추게 설정해라

우리가 훈련을 어디서 멈춰야 하는가??
어느정도 값이 되었을 때를 설정하고
'''