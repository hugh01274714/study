import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np 

#1. 데이터 (정제된 데이터)
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1)) # 데이터 세트에 하나를 넣어서 하나를 얻겠다.

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') # mse = 평균 제곱 에러 > loss 값은 작으면 좋다. / optimizer = 'adam' -> loss의 mse를 감축시키는 역할을 한다.

model.fit(x, y, epochs=2500, batch_size=2) # epochs = 훈련량 / batch_size = 몇개씩 잘라서 훈련 시킬 것인지, 배치가 작을 수록 훈련이 잘된다. -> 데이터 양이 많을수록 오래걸리고 오류가 생김

#4. 평가, 예측 
loss = model.evaluate(x,y) # fit 이후 저장되어있는 훈련값(데이터)
print('loss :', loss)
result = model.predict([4])
print('4의 예측값 :', result)

'''
최적의 결과 값 기입하기
loss : 0.0
4의 예측값 : [[4.]]
'''

# epochs=5000, batch_size=1 일때 결과 값이 가장 좋았다. 
