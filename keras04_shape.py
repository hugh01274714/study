import numpy as np
a1 = np.array([[1,2],[3,4],[5,6]])
a2 = np.array([[1,2,3],[4,5,6]])
a3 = np.array([[[1],[2],[3]],[[4],[5],[6]]])
a4 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
a5 = np.array([[[1,2,3],[4,5,6]]])
a6 = np.array([1,2,3,4,5])


print(a1.shape) # 3행 2열
print(a2.shape) # 2행 3열
print(a3.shape) # 2면 3행 1열
print(a4.shape) # 2면 2행 2열
print(a5.shape) # 1면 2행 3열
print(a6.shape) # (5, )

# a1=np.arange(6).reshape(2,3) 2차원 배열 변환
