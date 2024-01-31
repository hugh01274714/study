from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x =datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 49)

# 2. 모델구성
model = Sequential()
model.add(Dense(233, input_dim=10)) 
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

model.fit(x_train, y_train, epochs=1000, batch_size=2,
          validation_split=0.5)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

