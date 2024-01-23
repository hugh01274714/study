from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
x_train = x[:12]
x_test = x[12:15]
y_train = y[:12]
y_test = y[12:15]
x_val = x[15:]
y_val = x[15:]

'''
x_train = np.array(range(11))
y_train = np.array(range(11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])
'''

#2. 모델구성
model = Sequential()
model.add(Dense(233, input_dim=1)) 
model.add(Dense(144))
model.add(Dense(89))
model.add(Dense(55))
model.add(Dense(34))
model.add(Dense(21))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=200, batch_size=1,
          validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([17])
print('17의 예측값 :', y_predict)


'''
데이터를 받는 순간 3등분 해야한다. train_test_validation (cross validation이 나오기 전까지)
train = 내신(교과서 위주의) >> 머신이 한다.(fit에서)
>>> 컴퓨터 스스로 훈련하고 검증한다면, 성능이 훨씬 더 좋아진다.
>>>> 훈련데이터의 일부를 잘라서 검증데이터로 준비한다. 
>>>>> 훈련 - > 검증 (1epochs)
test = 수능 >> 사람이 한다.(evaluate에서, 버려지는 데이터) 

'loss'는 훈련에 과적합 되어있기 때문에 val_loss를 더 신뢰할 수 있다. >> 훈련에 대한 검증셋으로 신뢰성을 확인한다.


'''