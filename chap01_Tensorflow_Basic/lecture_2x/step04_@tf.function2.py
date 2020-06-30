# -*- coding: utf-8 -*-
"""
step04_@tf.function2.py

- Tensorflow2.0 특징
  3. @tf.function 함수 장식자(데코레이터)
     - 여러 함수들을 포함하는 main 함수
     - 피호출 함수는 함수 장식자를 안붙여도 된다.
"""

import tensorflow as tf

# model 생성 함수
def linear_model(x):
    return x * 2 + 0.2 # 회귀방정식

# model 오차 함수
def model_error(y, y_pred):
    return y - y_pred # 오차

# model 평가 함수 : main
@tf.function
def model_evaluation(x, y):
    y_pred = linear_model(x)       # 함수 호출
    error = model_error(y, y_pred) # 함수 호출
    return  tf.reduce_mean(tf.square(error)) # mse

# x, y data 생성
X = tf.constant([1,2,3], dtype = tf.float32)
Y = tf.constant([2,4,6], dtype = tf.float32)

mse = model_evaluation(X,Y)
print("MSE = %5f" % (mse)) # MSE = 0.040000