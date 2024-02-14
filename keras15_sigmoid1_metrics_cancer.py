import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (569, 30) (569,)
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 46)
'''
print(y_test[:10])
print(y)
#print(y[:10])
print(np.unique(y)) # [0 1], 분류값에서 고정이 되는 값
'''

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(50, activation= 'linear', input_dim=30))
model.add(Dense(70, activation= 'sigmoid')) 
model.add(Dense(50, activation= 'linear'))
model.add(Dense(50, activation= 'linear'))
model.add(Dense(50, activation= 'sigmoid'))
model.add(Dense(1, activation= 'sigmoid')) # 결과치가 0과1이나오게 하기 위해 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함

model.fit(x_train, y_train, epochs=100, batch_size=5,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)


'''
@ 회귀에서만 적용 가능하다.
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)

loss : [0.6669519543647766, 0.640350878238678] 
0.6669519543647766 : loss // 전체 최종 loss 값이며, 앞 쪽에 위치는 고정이다.
0.640350878238678 : metrics=['accuracy']

metrics=['accuracy']는 훈련에 현재 상황만 출력해서 보여주는 것이다. 
>> 훈련에 영향을 미치지 않는다. , 다른 지표를 또 사용할 수 있다.
리스트'[]'는 두 개 이상일 때 사용한다. >> 중괄호 사용은 이유가 있다. 

------------------------------------------------------------------------

@ 오전 정리 
'이진 분류'는 0과1을 결과값으로 도출한다. 
sigmoid >> 어떤 x 값을 넣더라도 0에서 1사이의 값을 도출한다.
최종 아웃풋 레이어에 activation은 sigmoid를 써야한다. 
loss는 binary cross entropy를 쓴다.

activation을 바꿈에 따라 성능차가 생기기 시작한다. 
디폴트 값인 'linear'로만 했을 경우 loss값이 튀면서 안좋아진다. 
metrics=['accuracy']는 평가에 영향을 미치지 않지만, 몇 개가 맞아는지에 대해서 수치로 나타내준다.
결과값은 loss와 accuracy 값을 리스트로 나타낸다. 

------------------------------------------------------------------------

@ 오후 시작 

'이진분류'는 '다중분류'이다. 

1. 회귀
'손실'='loss'='cost' 을 작게 잡는다. : 실제 데이터와 선의 거리를 가깝게 하다.

2. 이진 분류 : 데이터의 y값이 0과 1사이 >> 모델 구성 마지막 sigmoid >>> loss는 binary cross entropy

3. 다중 분류 : y값이 0,1,2 / 0,1,2,3 등으로 구성되어 있을 때 >> 마지막 모델 구성은 softmax
                                                        >>> loss가 categorical_crossentropy
soft max : 라벨의 갯수만큼 빼줬다. 총합은 1이다. 다중 분류 모델 구성에서 가장 마지막에 사용
다중분류는 마지막 모델 구성 시에는 라벨의 갯수와 동일하게 구성해야 한다. 
다중분류에서는 마지막 레이어의 activation은 softmax 
다중분류는 y값을 One-Hot Encoding 해준다.

데이터의 shape를 찍어보고, y값을 확인하며 다중분류 여부를 결정한다. 
y값을 라벨 수에 맞춰서 OHE 해주어야 한다. 
그 이후 모델을 만들고 나서 마지막 레이어에 softmax 해주어야 한다. 



'''
