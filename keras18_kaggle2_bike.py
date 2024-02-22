import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
# sqrt : 제곱근

#1. 데이터 
path = 'D:\\_data\\kaggle\\bike\\'   # '..'의 뜻은 이전 단계이다. / '.'은 현재 단계 >> 여기선 STUDY 폴더
train = pd.read_csv(path+'train.csv')  
# print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
# print(test.shape)    # (6493, 9)
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
# print(submit.shape)     # (6493, 2)
print(submit_file.columns)    # ['datetime', 'count']


# print(train.info())
print(test_file.describe())   
# 'object': 모든 자료형의 최상위형, string형으로 생각하면 된다.   
# 0   datetime    10886 non-null  object는 수치화 할 수 없다. >> 수치화 작업을 해주어야 한다. 
# print(type(train)) # <class 'pandas.core.frame.DataFrame'>
# print(train.describe()) # mean 평균, std 표준편차, min 최소값, 50% 중위값, holiday는 0과 1(휴일), 
# print(train.columns) 
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'], 
# dtype='object')
# print(train.head(3))
# print(train.tail())


x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 

# print(x.columns) 

'''
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed'],
      dtype='object')
'''   
# print(x.shape)      # (10886, 8)
y = train['count']
# print(y.shape)      # (10886,)
# print(y)

# 로그변환
y = np.log1p(y)

# plt.plot(y)
# plt.show()
# 데이터가 우상향하는 것처럼 한쪽으로 치우친 경우에는 로그 변환 시켜준다. 
# 로그 변환의 가장 큰 문제 : 0이라는 숫자가 나오면 안된다. 
# >> 안나오게 하려면?? 로그하기 전에 1을 더해준다. 


x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.9, shuffle=True, random_state = 666)


#2. 모델구성
model = Sequential()
model.add(Dense(233, input_dim=8)) 
model.add(Dense(144))
model.add(Dense(89))
model.add(Dense(55))
model.add(Dense(34))
model.add(Dense(144))
model.add(Dense(89))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='auto',
                   verbose=1, restore_best_weights=False)

model.fit(x_train, y_train, epochs=1000, batch_size=0,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print ('r2 :', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)


'''
@ 로그 변환 전
loss :  24541.5703125
r2 : 0.2511590856445959
RMSE :  156.6575036311734
@ 로그 변환 후
loss :  1.5552915334701538
r2 : 0.23641423053371247
RMSE :  1.2471133862193395
'''

##################### 제출용 제작 ####################
results = model.predict(test_file)

submit_file ['count'] = results

# print(submit_file[:10])

submit_file.to_csv(path + 'kovtzz.csv', index=False) # to_csv하면 자동으로 인덱스가 생기게 된다. > 없어져야 함



'''
# 과제 
중위값과 평균 값의 차이 

1) 평균 값 : 데이터를 모두 더한 후 그 전체자료 갯수로 나눈 값을 말함
2) 중위 값(중앙 값,median) : 데이터 변량을 크기의 순서로 늘어 놓았을 때, 중앙에 위치하는 값을 의미

로그(log)는 어떤 수를 나타내기 위해 고정된 밑을 몇 번 곱하여야 하는지를 나타냄
지수(exponential)는 거듭제곱의 지수를 변수로 하고, 정의역을 실수 전체로 정의하는 '초월함수', '로그함수'의 역함수

'''