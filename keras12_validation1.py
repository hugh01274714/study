from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 

#1. 데이터
x_train = np.array(range(11))
y_train = np.array(range(11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])

#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=1,
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

validation이 유효하게 사용될려면, 모델 설정을 잘해야 한다. 
'''