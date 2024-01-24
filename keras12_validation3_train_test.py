from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

# train_test_split로 나누시오 (10개 3개 3개) 
# 처음 16개를 train과 test로 나누고, 이후에 test와 val로 나눈다.
x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.625,shuffle=False,random_state=66) #shuffle=True는 디폴트 값이다.
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, 
                train_size=0.5, random_state=42)

print(x_train)
print(x_test)
print(x_val)


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


