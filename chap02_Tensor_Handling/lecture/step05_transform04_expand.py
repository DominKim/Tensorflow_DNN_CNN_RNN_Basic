'''
expand_dims <-> squeeze
 - tensor에 축 단위로 차원을 추가하는 함수 
'''

import tensorflow as tf
const = tf.constant([1,2,3,4,5]) # 1차원 

print(const)
print(const.shape) # (5,)

d0 = tf.expand_dims(const, axis=0) # 행 축(열 위치) 2차원 
print(tf.shape(d0)) # [1 5]
print(d0) # [[1 2 3 4 5]], shape=(1, 5)
    
d1 = tf.expand_dims(const, axis=1) # 열 축(행 위치) 2차원 
print(tf.shape(d1)) # [5 1]
print(d1)
'''
[[1]
 [2]
 [3]
 [4]
 [5]], shape=(5, 1)
'''

# 행렬곱에서 차수 불일치 문제 해결
x_data = tf.constant([10,20])
x_data.shape # [2]

y_data = tf.constant([[1, 2, 3], [4, 5, 6]])
y_data.shape # [2, 3]

x_data_dim = tf.expand_dims(x_data, axis = 0)
x_data_dim.shape # [1, 2]

mat = tf.matmul(x_data_dim, y_data)
print(mat)

