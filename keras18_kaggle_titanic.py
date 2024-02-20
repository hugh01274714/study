import numpy as np
import pandas as pd

path = './_data/titanic/'
train = pd.read_csv(path + 'train.csv', index_col=0, header=0) # header=헤더 변경, index_col=0 > 인덱스 수정 >> default 일때 확인

# train = train.drop(['Cabin'],1, inplace=False)  # (712, 7)
# train = train.dropna()
# y= train['Survived']
# x = train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'],1, inplace =True)
# x = pd.get_dummies(train)

print(train[:5])
print(train.shape) # (891, 11)  
# print(np.unique(y))

'''
from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(x, y)
'''

test = pd.read_csv(path + 'test.csv', index_col=0, header=0)
gender_submission = pd.read_csv (path+'gender_submission.csv'
                                 , index_col=0, header=0)
# print(test.shape)        # (418, 10)
# print(gender_submission.shape)  # (418, 1)
print(train.info())
print(test.describe())   

# ids = test[['PassengerId']]
# test.drop(['PassengerId', 'Name', 'Ticket','Cabin'],1,inplace=True)
# test.fillna(2, inplace=True)
# test = pd.get_dummies(test)
# # print(test)

# print(test.shape)  # (418, 10)
# print(np.unique(test))
# predictions = dtc.predict(test)
# results = ids.assign(Survived = predictions)

# results = ids.assign(Survived = predictions) # assign predictions to ids
# results.to_csv("titanic-results.csv", index=False) 



'''
gender_submission = pd.read_csv (path+'gender_submission.csv')
train.head()
'''
'''
train = pd.read_csv('./_data/titanic/train.csv')# csv는 간격이 ,로 되어 있고 엑셀은 셀로 구분되어 있다. 
test = pd.read_csv('./_data/titanic/test.csv')
gender_submission = pd.read_csv('./_data/titanic/gender_submission.csv')
'''
'''
print(train)
print(train.shape)        # (891, 12)
'''
'''
print(test)
print(test.shape)        # (418, 11)
print(gender_submission)
print(gender_submission.shape)    # (418, 2)
# PassengerId : PassengerId는 '데이터'인가?? >> NO, 행에 대한 인덱스(행에 숫자를 넣은 것) 일 뿐이다. 
# '헤더'도 잘못 되어있다. >> 11개의 '컬럼'  // read_csv로 검색하면 
# survived 라는 컬럼이 y가 된다. 
'''