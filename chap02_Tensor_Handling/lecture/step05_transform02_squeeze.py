'''
squeeze
 - 차원의 size가 1인 경우 제거
'''

import numpy as np
a = np.zeros( (1,2) ) # 2차:행/열
print(a)
print(np.squeeze(a)) # 1개 차원 삭제

b = np.zeros( (1,2,1) ) # 3차:면/행/열
print(b)
print(np.squeeze(b)) # 2개 차원 삭제

c = np.zeros( (1,2,1,3) ) # 4차원 
print(c)
print(np.squeeze(c)) # 2개 차원 삭제 


import tensorflow as tf
print("\ntensorflow")
t = tf.zeros( (1,2,1,3) )
t.shape

print(tf.squeeze(t)) # shape=(2, 3)

print(tf.squeeze(t).shape) # (2, 3)






