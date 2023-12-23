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
model.add(Dense(1000, input_dim=1)) # 데이터 세트에 하나를 넣어서 하나를 얻겠다.
model.add(Dense(500))
model.add(Dense(600))
model.add(Dense(150))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # mse = 평균 제곱 에러 > loss 값은 작으면 좋다. / optimizer = 'adam' -> loss의 mse를 감축시키는 역할을 한다.

model.fit(x, y, epochs=30, batch_size=3) # epochs = 훈련량 / batch_size = 몇개씩 잘라서 훈련 시킬 것인지, 배치가 작을 수록 훈련이 잘된다. -> 데이터 양이 많을수록 오래걸리고 오류가 생김

#4. 평가, 예측 
loss = model.evaluate(x,y) # fit 이후 저장되어있는 훈련값(데이터)
print('loss :', loss)
result = model.predict([4])
print('4의 예측값 :', result)


'''
epochs=30으로 고정해서 하이퍼 파라미터 튜닝 : 히든 레이어를 수정(숫자변경, 레이어 추가/삭제 등)하여, 4가 도출할 수 있도록
'''

'''
model = Sequential()
model.add(Dense(300, input_dim=1)) 
model.add(Dense(150))
model.add(Dense(160))
model.add(Dense(105))


model.add(Dense(75))
model.add(Dense(1))
 
노드 값을 커졌다 작아졌다 2번 반복 했을 시 예측값 3.99까지 도출했음
'''