'''
index 리턴 
  1. argmin/argmax
   - 축 별 최소/최대 값의 index 반환 
  2. unique/setdiff1d
   - 중복제거/차집합 결과 index 반환  
'''

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함

# 1. argmin/argmax
a = tf.constant([5,2,1,4,3], dtype=tf.int32)
b = tf.constant([4,5,1,3,2])
c = tf.constant([[5,4,2], [3,2,4]]) # 2차원 

# dimension : reduce 차원(vector = 0) 
min_index = tf.arg_min(a, dimension=0) # 1차원 대상
max_index = tf.arg_max(b, dimension=0) # 1차원 대상 
max_index2 = tf.arg_max(c, dimension=1) # 2차원 대상
#
sess = tf.Session()
print(sess.run(min_index)) # 2
print(sess.run(max_index)) # 1
print(sess.run(max_index2)) # [0 2]


# 2. unique/setdiff1d

c = tf.constant(['a','b','a','c','b'])
# unique 
cstr, cidx = tf.unique(c)
print(sess.run(cstr)) # [b'a' b'b' b'c']
print(sess.run(cidx)) # [0 1 0 2 1]

# setdiff1d : [5,2,1,4,3] - [1,3,2]
d = tf.constant([1,3,2], dtype=tf.int32)
set_result, set_idx = tf.setdiff1d(a, d)
print(sess.run(set_result)) # [5 4]
print(sess.run(set_idx)) # [0 3]






