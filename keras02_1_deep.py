# import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np 

#1. 데이터 (정제된 데이터)
x = np.array([1,2,3])
y = np.array([1,2,3])
# 위 데이터를 훈련해서 최소의 loss를 만들어보자.

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) # 데이터 세트에 하나를 넣어서 하나를 얻겠다.
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # mse = 평균 제곱 에러 > loss 값은 작으면 좋다. / optimizer = 'adam' -> loss의 mse를 감축시키는 역할을 한다.

model.fit(x, y, epochs=50, batch_size=3) # epochs = 훈련량 / batch_size = 몇개씩 잘라서 훈련 시킬 것인지, 배치가 작을 수록 훈련이 잘된다. -> 데이터 양이 많을수록 오래걸리고 오류가 생김

#4. 평가, 예측 
loss = model.evaluate(x,y) # fit 이후 저장되어있는 훈련값(데이터)
print('loss :', loss)
result = model.predict([4])
print('4의 예측값 :', result)


'''

'''

#  
