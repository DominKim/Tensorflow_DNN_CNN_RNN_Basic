# -*- coding: utf-8 -*-
"""
step05_variable_type.py

Tensorflow 변수 유형
 1. 초기값을 갖는 변수 : Fetch  방식
    변수 = tf.Variable(초기값)
 2. 초기값이 없는 변수 : Feed 방식
    변수 = tf.placeholder(dtype, shape)
"""

import tensorflow.compat.v1 as tf # ver 2.x -> ver 1.x
tf.disable_v2_behavior() # ver2.x 사용

# 상수 정의
x = tf.constant(100.0)
y = tf.constant(50.0)

# 식 정의
add = tf.add(x,y) # 150 = 100 + 50

# 변수 정의
var1 = tf.Variable(add) # Fetch 방식 : 초기값
var2 = tf.placeholder(dtype = tf.float32) # Feed  방식 : 초기값(x)

# 변수 참조하는 식
mul = tf.multiply(x, var1)
mul2 = tf.multiply(x, var2)

with tf.Session() as sess:
    print("add =", sess.run(add)) # 식 실행 : add = 150
    sess.run(tf.global_variables_initializer()) # 변수 초기화(Variable, fetch 방식)
    print("var1 =", sess.run(var1)) # 변수 생성 : var1 = 150
    print("var2 =", sess.run(var2, feed_dict = {var2: [1.5, 2.5, 3.5]})) # 데이터 공급 : feed_dict = {공급받을 변수 : 값} : var2 = 150.0
    
    mul_re = sess.run(mul) # 상수(100)와 변수(150) 참조
    print("mul =", mul_re) # mul = 15000.0
    
    # feed 방식의 식 연산 수행
    mul_re2 = sess.run(mul2, feed_dict = {var2: [1.5, 2.5, 3.5]}) 
    # feed_dict = 지역변수이므로 매번 할당해야 한다.!!!
    print("mul2 =", mul_re2) 
    # mul2 = 15000.0(scala), mul2 = [150. 250. 350.](vector)
    
