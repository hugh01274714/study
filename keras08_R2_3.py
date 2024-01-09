

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
# 데이터 삭제 시
'''
x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.7, shuffle=True, random_state=66)
'''

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1)) 
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))

# 피보나치 수열 사용 시 r2스코어 : 0.8099961017434907 까지 도달

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

model.fit(x,y, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y) # loss가 훈련에 영향을 미치지 않는다. 
print('loss:', loss) 

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict) # r2는 정확도와 상응되는 지표, loss의 보조지표
print('r2스코어 :', r2)

# loss: 21.302291870117188
# r2스코어 : -1.0341711202291357

'''
plt.scatter(x,y) # 데이터를 점으로 흩뿌린다.
plt.plot(x, y_predict, color='red') # 연속된 선을 그려준다.
plt.show()
'''

# 위와 같은 선형은 상대적이다. 
# mse와 rmse 확인하기 
# 분류데이터 : 남자 or 여자, 개 or 고양이 >> 정확도를 찾는 것을 한다. 