#1. R2를 음수가 아닌 0.5 이하로 만들 것
#2. 데이터를 건들지 말 것
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4. batch_size = 1
#5. epochs는 100이상
#6. 히든 레이어의 노드는 10개 이상 1000개 이하
#7. train 70%


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))


x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.7, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim = 1))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

# 노드 갯수를 동일하게 넣고, 레이어를 늘렸을 때 -> r2 스코어가 낮아진다.

 
#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) 
print('loss:', loss) 

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

'''
model = Sequential()
model.add(Dense(300, input_dim = 1))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(1))

위와 같이 모델 구성 했을 때 r2 : 0.35923425867930003 까지 떨어졌다. 

model = Sequential()
model.add(Dense(400, input_dim = 1))
model.add(Dense(400))
model.add(Dense(400))
model.add(Dense(400))
model.add(Dense(400))
model.add(Dense(400))
model.add(Dense(400))
model.add(Dense(400))
model.add(Dense(1))
위와 같이 모델 구성 했을 때 r2 : -1.1671155502136599 까지 떨어졌다. 

>> but, 바로 다시 돌렸을 때 0.999로 바뀜..??..
>>> 왜 why? 훈련되었기 때문에


'''

