

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time # 시간에 대한 것들이 임포트 됨 


#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

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



#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

start = time.time() # 
model.fit(x,y, epochs=1000, batch_size=1, verbose=0)
end = time.time() - start
print('걸린시간:', end)

'''
verbose=0 > 없다. >> 걸린시간: 3.3190934658050537
verbose=1 > 전체가 다 보인다. >> 걸린시간: 4.431308031082153
verbose=2 > loss만 보인다. >> 걸린시간: 3.3920116424560547
verbose=3~ > epochs 만 나온다. >> 걸린시간: 3.5945067405700684

4가지의 차이는 ?? 
컴퓨터의 연산은 빠르기 때문에, 인간이 인식 할 수 있도록 표시 후 '딜레이'를 발생시킨다.

'''
'''
#4. 평가, 예측
loss = model.evaluate(x,y) 
print('loss:', loss) 

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict) 
print('r2스코어 :', r2)
'''
