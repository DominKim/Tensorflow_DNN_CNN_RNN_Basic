# -*- coding: utf-8 -*-
"""
python code vs tensorflow code
"""

# python : 직접 실행 환경 / tensorflow : 간접 실행 환경
x = 10
y = 20
z = x + y
print("z = %d" % z)

# import tensorflow as tf # version  2.0
# 마이그레이션 [Migration]
import tensorflow.compat.v1 as tf # ver 2.x -> ver 1.x
tf.disable_v2_behavior() # ver2.x 사용

print(tf.__version__)

'''프로그램 정의 영역'''
# 상수 : 수정불가 
x = tf.constant(10) # 상수 정의
y = tf.constant(20) # 상수 정의
print(x, y)
'''
Tensor("Const_2:0", shape=(), dtype=int32)   shape = () : 0차원 스칼
Tensor("Const_3:0", shape=(), dtype=int32)
'''
# 식 정의
z = x + y
print("z = ", z)
# z =  Tensor("add_1:0", shape=(), dtype=int32)

# session 객체 생성
sess = tf.Session() # 상수, 변수, 식 -> device(CPU, GPU, TPU) 할당

'''프로그램 실행 영역'''
print("x = ", sess.run(x)) # x = 10
print("y = ", sess.run(y)) # y = 20
# sess.run(x, y) error

# 하나의 세션안에서 여러개의 상수나 식 출력
# 기존의 정의 되어있는 변수에 출력 값을 할당 하면 ovrewriting  된다.
x_val, y_val = sess.run([x, y])
print(x_val, y_val) # 10 20

print("z = ", sess.run(z)) # x,y 상수 참조 -> 연산

# 객체 닫기
sess.close()

