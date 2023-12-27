import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# mlp 뜻 : 멀티 레이어 퍼셉트론

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,
             1.6,1.5,1.4,1.3]])
# x= np.transpose(x)
x= x.T
y = np.array([11,12,13,14,15,16,17,18,19,20])

#2. 모델구성

model = Sequential()
model.add(Dense(100, input_dim=2)) 
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam') 

model.fit(x, y, epochs=50, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss :', loss)
y_predict = model.predict([[10,1.3]])
print('[10,1.3]의 예측값 :', y_predict)

'''
배열을 변환할 때 사용, 
x = np.transpose(x) >>but, 데이터의 순서가 바뀔 수 있다.
x = np.reshape(10,2) >> 데이터의 순서가 바뀌지 않는다.
x=x.T

y_predict = model.predict([10,1.3]) > ([[10,1.3]]) >>> x의 input 디멘션 컬럼(열, 특성,피처)의 개수와 같다.
*열 우선, 행 무시

다층 퍼셉트론 (multu-layer percepton,MLP) 뜻 : 퍼셉트론으로 이루어진 층 여러개를 
순차적으로 붙여놓은 형태
정방향 인공신경망(feed-forward deep neural network, FFDNN) 이라고 부르기도 함.
퍼셉트론이란?? : 다수의 신호를 입력받아 하나의 신호를 출력하는 알고리즘 [y=wx+b]

>>> 다수의 신호를 입력받아 하나의 신호를 출력하는 알고리즘들로 이루어진 층 여러개를 순차적으로 
붙여놓은 형태

'''