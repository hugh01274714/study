'''
R2(R2 score)는 무엇인가??? :
회귀식의 성능을 평가하는 지표로 사용하는 결정계수 값, 예측한 모델이 얼마나 실제 데이터를 설명하는지를 나타낸다
평균값으로 예측을 했을 때의 오차이ㅡ 제곱의 합과 실제 우리가 예측한 모델의 오차의 제곱의 합을 비교 후 1에서 뺀 형태로
나타낸다.
모델이 모든 데이터를 완벽하게 설명하면 실제 우리가 예측한 모델의 값(면적)이 0이 되고 R2 값은 '1'을 갖게된다.
>> loss만으로 판단하기 어려울 때 사용한다. 
>>> 두 개를 같이 사용
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,9,8,12,13,17,12,14,21,14,11,19,23,25])
# 데이터 삭제 시
x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.7, shuffle=True, random_state=66)


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

model.fit(x_train,y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) # loss가 훈련에 영향을 미치지 않는다. 
print('loss:', loss) 

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # r2는 정확도와 상응되는 지표, loss의 보조지표
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