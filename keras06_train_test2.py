from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

### 과제 ###
# train과 test 비율을 8:2으로 분리하시오. >> 리스트의 슬라이싱으로 자르기  

x_train = x[:8]
x_test = x[8:]
y_train = y[:8]
y_test = y[8:]

### 완료 후 모델>평가,예측까지###
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
model.fit(x_train, y_train, epochs=100, batch_size=1) 

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)
result = model.predict([114])
print('11의 예측값 :', result)


'''

@ 리뷰
실질적으로 범위 안에 있는 것은 잘 맞지만, 그 외에 데이터는 잘 맞지 않는다. 
'도박사의 오류'에 의하여, 
8:2로 나누었을 때 test에 속하는 2는 훈련 시 버려지는 데이터이다.
그렇기 때문에 훈련 부분은 전체 데이터를 넣어야한다.
전체 데이터에서 일정 크기를 랜덤하게 빼서 테스트한다.

'''