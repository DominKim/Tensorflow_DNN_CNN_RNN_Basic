# -*- coding: utf-8 -*-
"""
step04_regression_formula2.py

 다중선형회귀방정식 : 행렬곱 이용
  - X(N) -> Y
  - y_pred = X1 * a1 + X2 * a2 + ... XN * aN + b
  - y_pred = tf.matmul(X, a) + b
"""

import tensorflow as tf

# X, Y변수 정의
X = [[1.0, 2.0]] # [1, 2] : 입력
Y = 2.5 # 정답

# a, b 변수 정의 : 수정 가능(난수 상수)
a = tf.Variable(tf.random.normal([2, 1]))# 기울기 : [2, 1]
b = tf.Variable(tf.random.normal([1]))

# model 식 정의
y_pred = tf.matmul(X, a) + b
# y_pred = tf.math.add(tf.matmul(X, a), b)
# print(y_pred)
# tf.Tensor([[-3.1874497]], shape=(1, 1), dtype=float32)
'''
tf.matmul(X, a) : 행렬곱
 1. X, a -> 행렬
 2. X 열수 == a 행수
'''

# model error
err = Y - y_pred

# loss function : 손실 반환
loss = tf.reduce_mean(tf.square(err))

print("최초 기울기(a)와 절편(b)")
print("a = {}, b = {}".format(a.numpy(), b.numpy()))

print("model error =", err.numpy())
print("loss function =", loss.numpy())
'''
1차 실행
최초 기울기(a)와 절편(b)
a = [[0.39862967]
 [0.9162106 ]], b = [-0.37992278]
model error = [[0.6488718]]
loss function = 0.42103457
'''

'''
2차 실행
최초 기울기(a)와 절편(b)
a = [[-0.48958328]
 [ 0.944783  ]], b = [-0.8012545]
model error = [[1.9012718]]
loss function = 3.6148345
'''