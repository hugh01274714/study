from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))

# 랜덤하게 훈련과 test를 7:3
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle = True, random_state=66)
# train_size=0.7, shuffle = True >> 파라미터에 속한다. 
# print(x_test)   # [ 8 93  4  5 52 41  0 73 88 68]
# print(y_test)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1)) 
model.add(Dense(80))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=1) 

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)
result = model.predict([100]) 
print('11의 예측값 :', result)

'''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42) 랜덤하게 분류
사이킷 런에는 레거시안 머신러닝이 모두 들어가 있다. >> 성능이 꽤좋다. >>> 데이터 전처리 하는 함수들이 많이 들어가 있다. 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
** 랜덤 난수를 잡는 이유 확실히 이해하기 
무작위로 정해지는 수를 '난수'라고하며, 코딩을 할 때 거의 등장하는 단어가 
'Random'이었다. 최근에 존 폰 노이만이라는 과학자가 '의사난수'라고해서 일정한 규칙에 따라 
랜덤한 수를 만드는 방법을 개발했다.

but, 일정 규칙으로 아무런 규칙이 없는 수를 만든다는 것은 모순이기 때문에 컴퓨터가 
생성하는 난수는 '진정한 난수'가 아니라, 난수에 가깝다는 의미에서 '의사난수'라고 부른다.


'''