from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_train = np.array([1,2,3,4,5,6,7])
y_test  = np.array ([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1)) 
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=1) #1epoch에 7번 연산, 100epoch 시 700번, batch_size=2일 때 350번

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)
result = model.predict([11])
print('11의 예측값 :', result)


'''
훈련과 평가를 나누었다. 
평균적으로 데이터는 전체로 받고, 훈련과 데이터를 나누어야 한다.

'''