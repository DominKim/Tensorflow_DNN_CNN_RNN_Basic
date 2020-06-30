'''
선형대수 연산 함수
 tf.transpose : 전치행렬   
 tf.diag : 대각행렬 -> tf.linalg.diag(x)  
 tf.matrix_determinant : 정방행렬의 행렬식 -> tf.linalg.det(x)
 tf.matrix_inverse : 정방행렬의 역행렬 -> tf.linalg.inv(x)
 tf.matmul : 두 텐서의 행렬곱 -> tf.linalg.matmul(x, y)
 tf.eye : 단위 행렬 -> tf.linalg.eye(x) : 주 대각원소 1, 나머지 0
'''

import tensorflow as tf
import numpy as np

# 정방행렬 데이터 생성 
x = np.random.rand(2, 2) # 지정한 shape에 따라서  0~1 난수 
y = np.random.rand(2, 2) # 지정한 shape에 따라서  0~1 난수 

tran = tf.transpose(x) # 전치행렬
dia = tf.linalg.diag(x) # 대각행렬 
mat_deter = tf.linalg.det(x) # 정방행렬의 행렬식  
mat_inver = tf.linalg.inv(x) # 정방행렬의 역행렬
mat = tf.linalg.matmul(x, y) # 행렬곱 반환 
eye = tf.linalg.eye(3) # 단위행렬

print(x)
print(tran)  
print(dia) 
print(mat_deter)
print(mat_inver)
print(mat)
print(eye)
'''
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
'''

# 단위행렬 -> one-hot encoding
import numpy as np

a = [0, 1, 2]
encoding = np.eye(len(a))[a]
print(encoding)
'''
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
'''

# tf.multiply vs tf.matmul
'''
tf.multiply : 브로드캐스트 
 - y_pred = X * a
tf.matmul : 행렬곱
 - y_pred = X1 * a1 + X2 * a2
'''










