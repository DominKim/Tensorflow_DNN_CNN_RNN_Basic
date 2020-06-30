# -*- coding: utf-8 -*-
"""
step02_function_replace.py

- Tensorflow2.0 특징
 2. 세션 대신 함수 
  - ver2.0 : python 함수 사용 권장
  - API 정리 : tf.placeholder() 삭제 : 함수 인수 대체
              tf.random_uniform -> tf.random.uniform
              tf.random_normal -> tf.random.normal
"""

import tensorflow as tf

''' step07_variable_feed.py -> ver2.0'''

'''
# 변수 정의
a = tf.placeholder(dtype = tf.float32) # shape 생략 : 가변형
b = tf.placeholder(dtype = tf.float32) # shape 생략 : 가변형

c = tf.placeholder(dtype = tf.float32, shape = [5]) # 고정형 : 1d
d = tf.placeholder(dtype = tf.float32, shape = [None,3]) # 고정형 : 2d(행 가변)
# 행의 길이를 모를 때 : None

c_data = tf.random_uniform([5]) # 상수 정의 ! 

# 식 정의
mul = tf.multiply(a, b)
add = tf.add(mul, 10)
c_calc = c * 0.5 # 1d(vector) * 0d(scala)
'''

def mul_fn(a,b): # tf.placeholder() -> 인수 대체
    return tf.multiply(a,b)

def add_fn(mul):
    return tf.add(mul, 10)

def c_calc_fn(c):
    return tf.multiply(c, 0.5)

# data 생성
a_data = [1.0, 2.5, 3.5]
b_data = [2.0, 3.0, 4.0]

mul_re = mul_fn(a_data, b_data)
print("mul =", mul_re.numpy()) # mul = [ 2.   7.5 14. ]

print("add ={}".format(add_fn(mul_re))) # add =[12.  17.5 24. ]

# tf.random_uniform( ) # vear1.x
c_data = tf.random.uniform(shape = [3,4], minval=0, maxval=1) # ves2.


print("c_calc_function :", c_calc_fn(c_data).numpy())
'''
c_calc_function : [[0.2759002  0.00739062 0.4466113  0.23543161]
 [0.4079733  0.10840893 0.40078646 0.35052258]
 [0.28766894 0.41822457 0.42808652 0.15726125]]
'''
