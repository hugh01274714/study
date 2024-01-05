from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,17,8, 14,21, 9, 6,19,23,21])

x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim = 1))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train, epochs=1000, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) # loss가 훈련에 영향을 미치지 않는다. 
print('loss:', loss) 

y_predict = model.predict(x)

plt.scatter(x,y) # 데이터를 점으로 흩뿌린다.
plt.plot(x, y_predict, color='red') # 연속된 선을 그려준다.
plt.show()

# 위와 같은 선형은 상대적이다. 
# mse와 rmse 확인하기 
# 분류데이터 : 남자 or 여자, 개 or 고양이 >> 정확도를 찾는 것을 한다. 