from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt # 그림 그리는 것
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
datasets = load_boston()

#1. 데이터 

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.8, shuffle=True, random_state=1) #random_state=1로 했을 때 0.74 최대 값이 나옴

# 부동 소수점 연산?? e-03 앞에 0이 3개 ex) 6.3200e-01일 때 0.1을 곱하면 된다.  +01이면 10을 곱하면 된다.
# (506, 13)
# (506,) = 스칼라 506개, 벡터가 1개 >> 결과치 1개  

#2. 모델구성
model = Sequential()
model.add(Dense(233, input_dim=13)) 
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

# loss: 25.594913482666016
# r2스코어 : 0.7410139537683371

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) 
print('loss:', loss) 

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


#0.8 넘기면 컬럼에 대해 분석해라.
#loss와 r2는 비례적이지 않다. loss가 실제적으로 신뢰성이 더 있다.