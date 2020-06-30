'''
step02_convert_tensors.py

python 객체 -> Tensor 객체 변환 
'''
import numpy as np
import tensorflow as tf

# 1. numpy(1차) -> tensor 변환 
np_1d = np.array([1.3,1,4.0,23.99])

# 2. tensor 형식으로 변환 
tensor_np1 = tf.convert_to_tensor(np_1d, dtype=tf.float64)

print( tensor_np1 )
print( tensor_np1[2] )
    
# 1. numpy(2차) -> tensor 변환
np_2d=np.array([[1,2,3,4],[4,5,6,7],[8,9,10,11]])

# 2. tensor 형식으로 변환 
tensor_np2 = tf.convert_to_tensor(np_2d, dtype=tf.int32)

print(tensor_np2)
print(tensor_np2[2, 3]) 
print(tensor_np2[1:3, 1:3])  
  
 
    