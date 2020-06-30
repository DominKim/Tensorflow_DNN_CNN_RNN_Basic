'''
step01_shape_rank_size.py

Tensor의 차원(rank)과 shape 이해
 1. tensor shape
 2. tensor rank
 3. tensor size
 4. tensor reshape 
'''

import tensorflow as tf
print(tf.__version__) # 2.0.0

scalar = tf.constant(1234) 
vector = tf.constant([1,2,3,4,5])
matrix = tf.constant([ [1,2,3], [4,5,6] ])
cube = tf.constant([[ [1,2,3], [4,5,6], [7,8,9] ]])


# 1. tensor shape : 차원 모양
print(scalar.get_shape()) 
print(vector.get_shape())
print(matrix.get_shape())
print(cube.get_shape())

print(scalar)
print(vector)
print(matrix)
print(cube)
  
# 2. tensor rank : 차원 갯수
print('\ntensor rank')
print(tf.rank(scalar)) 
print(tf.rank(vector)) 
print(tf.rank(matrix)) 
print(tf.rank(cube))

# 3. tensor size : tensor가 가진 원소의 갯수 반환
print('\ntensor size')
print(tf.size(scalar)) 
print(tf.size(vector)) 
print(tf.size(matrix)) 
print(tf.size(cube))

# 4. tensor reshape 
# print('\ntensor reshape')
cube_2d = tf.reshape(cube, (3, 3)) # 모양 변경 : 3d -> 2d 
print(cube_2d)
print(tf.rank(cube_2d))


