# -*- coding: utf-8 -*-
"""
step02_tf_variable_init.py
 - 변수 정의와 초기화
 
 - 상수 vs 변수
   상수 : 수정 불가, 초기화 필요 없음
   변수 : 수정 가능, 초기화 필요 있음
"""
import tensorflow.compat.v1 as tf # ver 2.x -> ver 1.x
tf.disable_v2_behavior() # ver2.x 사용

''' 프로그램 정의 정역'''
# 상수 정의
x = tf.constant([1.5, 2.5, 3.5], name = "x") # 1차원 : 수정 불가
print("x :", x)
# x : Tensor("x:0", shape=(3,), dtype=float32)

# 변수 정의 -> 초기화 후 할당해야 됨
y = tf.Variable([1.0, 2.0, 3.0], name = "y") # 1차원 : 수정 가능
print("y :", y)
# y : <tf.Variable 'y:0' shape=(3,) dtype=float32_ref>

# 식 정의
mul = x * y # 상수 * 변수
# graph = node(연산자:+-*/) + edge(데이터:x,y)
# tensor : 데이터의 자료구조(scala(0), vector(1), matrix(2), array(3), n-array)

sess = tf.Session()
# 변수 초기화 객체
init = tf.global_variables_initializer()

''' 프로그램 실행 영역 '''
print("x =", sess.run(x)) # 상수 할당 : x = [1.5 2.5 3.5]
sess.run(init) # 참조 -> 변수 값 초기화
# print("y =", sess.run(y)) # 변수 할당 : y = [1. 2. 3.]

# 식 할당
mul_re = sess.run(mul) # 식 할당(연산)
print("mul =", mul_re) # mul = [ 1.5  5.  10.5]
type(mul_re) # numpy.ndarray

print("sum =", mul_re.sum()) # sum = 17.0

sess.close() 
