# -*- coding: utf-8 -*-
"""
step03_regression_formula.py

단순 선형회귀방정식 : X(1) -> Y
 - y_pred = X * a(기울기) + b(절편)
 - err = Y - y_pred
 - loss function(cost function) : 정답과 예측치 간의 오차 반환 함수
   -> function(Y, y_pred) -> 오차(손실 or 비용) 반환 : MSE
"""
from sklearn.linear_model import LinearRegression

import tensorflow as tf # ver 2.0 사용

# a, b 변수 정의
a = tf.Variable(0.5) # 기울기
b = tf.Variable(1.5) # 절편

# 회귀모델 함수
def linear_model(X) : # X : 입력
    y_pred = (X * a) + b # 회귀방정식
    return y_pred

# 모델 오차(error)
def model_err(X, Y): # (입력, 정답)
    y_pred = linear_model(X)
    err = Y - y_pred
    return err

# 손실함수(loss function) : (정답, 예측치) -> 오차 반환(MSE)
def loss_function(X, Y):
    err = model_err(X, Y)
    loss = tf.reduce_mean(tf.square(err)) # MSE
    return loss

'''
오차 : MSE
Error  : 정답 - 예측치
Square : 부호(+), 페널티
Mean   : 전체 관측치의 오차 평균
'''


# X, Y 변수정의
X = tf.Variable(6.5) # 입력(input)
Y = tf.Variable(5.2) # 정답(output)

# 1차 식 : a = 0.5, b = 1.5
print("최초 기울기(a)와 절편(b)")
print("a = {}, b = {}".format(a.numpy(), b.numpy()))

print("model error =", model_err(X, Y).numpy())
print("loss function =", loss_function(X, Y).numpy())
'''
최초 기울기(a)와 절편(b)
a = 0.5, b = 1.5
model error = 0.4499998
loss function = 0.20249982
'''

# 2차 식 : a = 0.6, b = 1.2(기울기, 절편 수정)
print("2차 기울기, 절편 수정")
a.assign(0.6) # 기울기 수정(0.5 -> 0.6)
b.assign(1.2) # 절편 수정(1.5 -> 1.2)
print("최초 기울기(a)와 절편(b)")
print("a = {}, b = {}".format(a.numpy(), b.numpy()))

print("model error =", model_err(X, Y).numpy())
print("loss function =", loss_function(X, Y).numpy())

'''
2차 기울기, 절편 수정
최초 기울기(a)와 절편(b)
a = 0.6000000238418579, b = 1.2000000476837158
model error = 0.09999943
loss function = 0.009999885
'''

'''
[키워드 정리]
최적화된 model : 최적의 기울기와 절편 수정 -> 손실(loss) 0에 수렴
딥러닝 최적화 알고리즘 : GD, Adam -> 최적의 기울기와 절편 수정 역할
'''