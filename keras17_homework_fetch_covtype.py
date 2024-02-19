from matplotlib.pyplot import yscale
import numpy as np 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target
'''
print(np.unique(y)) # [1 2 3 4 5 6 7]
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) # (581012, 8)
'''

import pandas as pd 
y = pd.get_dummies(y)
print(x.shape,y.shape) 

'''
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y= ohe.fit_transform(y.reshape(-1,1))
print(y.shape) # (581012, 7)
'''  


# from tensorflow.keras.utils import to_categorical 
# >> 345679 일 때 categorical 을 사용한다면, 0에서부터 빈자리를 채워줘서, 0123456789로 만들어 준다.
# 0~7까지 돌렸을 때와 1~7까지 돌렸을 때의 차이점 
# y = to_categorical(y)

# print(np.unique(y))


x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 36)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(233, input_dim=54)) 
model.add(Dense(144))
model.add(Dense(55))
model.add(Dense(21))
model.add(Dense(7, activation= 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='auto',
                   verbose =1, restore_best_weights=False)

model.fit(x_train, y_train, epochs =100, batch_size=1000,
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('accuracy : ', loss[1])



# batch_size 디폴트 값은 32이다.
# 출처 : https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network


'''
to_categorical로 돌렸을 때에는 8개의 결과 값을 내고 있다. 
but, y는 [1~7]인 7개의 결과값을 갖고 있다. 
예를들면, 시험을 볼 때 4지선다 or 5지선다 라면, 4지 선다를 선택하는 것과 같이 
우리는 최적의 loss값을 도출 할 수 있는 방법을 찾아야 하기 때문에 
>import pandas as pd 
>y = pd.get_dummies(y)
를 사용해야 한다. 

'''





