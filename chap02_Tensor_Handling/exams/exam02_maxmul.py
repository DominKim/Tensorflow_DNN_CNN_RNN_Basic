'''
텐서플로우 행렬 연산 실습 
  - ppt.16 두번째 이미지 구현 
'''

import tensorflow as tf

x = tf.Variable([[10,20]])
y = tf.Variable([[1,2],[2,3]])
z = tf.linalg.matmul(x, y)
print(z.numpy())

x2 = tf.Variable([[1,2], [3,4]])
y2 = tf.Variable([[1,2,3],[2,3,4]])
z2 = tf.linalg.matmul(x2,y2)
print(z2.numpy())