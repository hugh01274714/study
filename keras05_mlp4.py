import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)]) # (10,)
x = np.transpose(x) # 해야하는지 확인해보기 > 현재(10,) >> input_dim=1

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,
             1.6,1.5,1.4,1.3],
             [10,9,8,7,6,5,4,3,2,1]])
y = np.transpose(y)
print(y.shape)

#2. 모델구성

model = Sequential()
model.add(Dense(100, input_dim=1)) 
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(3))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam') 

model.fit(x, y, epochs=50, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss :', loss)
y_predict = model.predict([[9]])
print('[9]의 예측값 :', y_predict)

'''
이미 훈련이 되어 있는 데이터이기 때문에 예측값이 잘 나올 수 밖에 없다.
model.evaluate(x,y) >> 공정한 연산인가??
loss 값은 신뢰할 수 있는가?? 
NO! -> 이미 연산되어 있는 값이기 때문에

>>> 개선을 해야한다. 
훈련데이터는 [model.fit(x, y, epochs=50, batch_size=3)] 에서 시키고, 나머지 30%는 [loss = model.evaluate(x,y)]에서 쓴다.

'''