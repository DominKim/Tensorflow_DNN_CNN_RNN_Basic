'''
reshape
 - tensor의 모양 변경 
'''

import tensorflow as tf
import numpy as np

print("numpy")
a = np.zeros([2,3])
print(a.reshape([1,6]))

print("\ntensorflow")
t = tf.zeros([2,3])
print(t)

print(tf.reshape(t, [1,6]))  # 2 -> 2

print(tf.reshape(t, [3,2]))

print(tf.reshape(t, [2,1,3])) # 2차원 -> 3차원
