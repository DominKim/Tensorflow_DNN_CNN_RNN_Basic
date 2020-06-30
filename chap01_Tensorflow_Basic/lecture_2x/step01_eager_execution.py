# -*- coding: utf-8 -*-
"""
step01_eager_execution.py

- Tensorflow2.0 특징
 1. 즉시 실행(eager execution) 모드
  - session object 없이 즉시 실행 환경(auto graph)
  - python 실행 환경과 동일함
  - API 정리 : tf.global_variables_initializer() 삭제
"""
import tensorflow as tf # ver 2.0
print(tf.__version__) # 2.0.0


# 상수 정의
a = tf.constant([[1,2,3], [1.0, 2.5, 3.5]]) # [2.3]
print("a :\n",a)
'''
a :
 tf.Tensor(
[[1.  2.  3. ]
 [1.  2.5 3.5]], shape=(2, 3), dtype=float32)
'''
print(a.numpy()) # 실제 data 추출

# 식 정의 : 상수 참조 -> 즉시 연산
b = tf.add(a, 0.5)
print("b :")
print(b)

# 변수 정의
x = tf.Variable([10,20,30])
y = tf.Variable([1,2,3])
print("x =", x.numpy())
print("y =", y.numpy())

mul = tf.multiply(x, y)
print("mul =", mul) # mul = tf.Tensor([10 40 90], shape=(3,), dtype=int32)
print("mul =", mul.numpy()) # mul = [10 40 90]

# python code -> tensorflow 즉시 실행
x = [[2.0, 3.0]] # [1, 2]
a = [[1.0], [2.5]] # [2, 1]

# 행렬곱 연산
mat = tf.matmul(x, a)
print("matrix multiply = {}".format(mat))
# .format(x) 형식으로 출력하면 vector값만 반환 한다. == .numpy
# matrix multiply = [[9.5]]

''' step02_tf_variable_init -> ver 2.0'''
''' 프로그램 정의 정역'''
print("~~ 즉시 실행 ~~")
# 상수 정의
x = tf.constant([1.5, 2.5, 3.5], name = "x") # 1차원 : 수정 불가

# 변수 정의 -> 초기화 후 할당해야 됨
y = tf.Variable([1.0, 2.0, 3.0], name = "y") # 1차원 : 수정 가능

# 식 정의
mul = x * y # 상수 * 변수

''' 프로그램 실행 영역 '''
print("x =", x.numpy()) # 상수 할당 : x = [1.5 2.5 3.5]
print("y =", y.numpy()) # 변수 할당 : y = [1. 2. 3.]

# 식 할당
print("mul =", mul.numpy()) # mul = [ 1.5  5.  10.5]

