from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

# 
x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.8125,shuffle=True,random_state=66) # 13개, 3개로 나눔 

# x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, 
                # train_size=0.5, random_state=42)



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
        #   validation_data=(x_val, y_val))
        validation_split=0.3) # train 데이터의 30%를 valdation을 사용하겠다. 
# >> fit에 영향을 미치지 않지만, 훈련의 방향성을 잡아준다.
# >>> 성능이 더 좋다.

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([17])
print('17의 예측값 :', y_predict)


