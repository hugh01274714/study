from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes # diabete 당뇨병


#1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.6, shuffle=True, random_state=49)



#2. 모델구성
model = Sequential()
model.add(Dense(13, input_dim=10)) 
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train, epochs=200, batch_size=1) #이것 만 훈련이다.
# print(x.shape, y.shape) # (442,10) , (442, )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) # 훈련에 영향을 미치지 않는다.
print('loss :', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)


# 과제 
# R2 : 0.62 이상

'''
@ 리뷰 

우리는 머신에게 정제된 x,y(데이터) 를 준다. >> 머신은 주어진 데이터로 최소의 'loss'를 찾는다.
>>> 최적의 weight를 찾는 것이 목적
x와 y는 1개 이상으로 설정할 수 있다. 
'loss'는 상대적 지표이기 때문에 'r2'를 함께 사용한다.
'경사 하강법 (Gradient Decent)'
loss는 컴파일에 있으며, optimizer(현재는 adam사용함) 가 loss를 최적화 시킨다. 
PCA : 주성분 분석 >> '특성'을 건들 때 사용
하이퍼 파라미터 튜닝이 진행 되는 부분은 '히든 레이어'이다.

!! 정제된 데이터를 준비하지 않으면, 최적의 결과 값을 도출할 수는 없다.
train_test
전체 데이터 중에 test 데이터는 날린다고 볼 수있다.
날리면서까지 train_test를 나누는 이유는 내신과 모의고사를 예를 들 수 있다. 
내신은 교과서의 특정 페이지만 공부하고, 특정 페이지에서만 나오는 것을 시험보기 때문에,

모의고사를 볼 경우  
[과적합을 막기 위해]
평가 데이터는 훈련에 영향을 미치지 않는다.

train : 교과서
test : 수능

'''