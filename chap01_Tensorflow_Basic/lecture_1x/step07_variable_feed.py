# -*- coding: utf-8 -*-
"""
step07_variable_feed.py

 2. 초기값이 없는 변수 : Feed 방식
    변수 = tf.placeholder(dtype, shape)
    - dtype : 자료형(tf.float32, tf.int32, tf.string)
    - shape : 자료구조([n] : 1차원, [r,c] : 2차원, 생략 : 공급data 결정)
"""

import tensorflow.compat.v1 as tf # ver 2.x -> ver 1.x
tf.disable_v2_behavior() # ver2.x 사용


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

with tf.Session() as sess:
    # 변수 초기화 생략
    
    # 식 실행
    mul_re = sess.run(mul, feed_dict = {a:2.5, b:3.5}) # data feed
    print("mul =", mul_re) # mul = 8.75
    
    # 공급 data
    a_data = [1.0, 2.0, 3.5]
    b_data = [0.5, 0.3, 0.4]
    feed_data = {a : a_data, b : b_data}
    
    mul_re2 = sess.run(mul, feed_dict = feed_data)
    print("mul_re2 =", mul_re2) # mul_re2 = [0.5 0.6 1.4]

    # 식 실행 : 식 참조
    add_re = sess.run(add, feed_dict = {a : a_data, b : b_data}) # mul +10
    print("add_re =", add_re) # add_re = [10.5 10.6 11.4]
    
    c_data_re = sess.run(c_data) # 상수 생성
    print(c_data_re)
    # [0.56301475 0.7152989  0.23478723 0.65178645 0.3032875 ]
    c_calc_re = sess.run(c_calc, feed_dict = {c:c_data_re}) 
    # 상수는 정의 후 생성 해야 할당이 된다.
    print("c_calc_re =", c_calc_re)
    # c_calc_re = [0.28150737 0.35764945 0.11739361 0.32589322 0.15164375]
    
    '''
    주의 : 프로그램 정의 변수와 리턴 변수명은 다르게 지정함
    '''