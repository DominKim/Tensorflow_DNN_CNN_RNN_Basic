# -*- coding: utf-8 -*-
"""
step04_tensorboard2.py

name_scope 이용 : 영역별 tensorflow 시각화
 - model 생성 -> model 오차 -> model 평가
"""

import tensorflow.compat.v1 as tf # ver 2.x -> ver 1.x
tf.disable_v2_behavior() # ver2.x 사용

# tensorboard 초기화
tf.reset_default_graph()

# 상수 정의 : X, a, b, Y
X = tf.constant(5.0, name = "x_data") # 입력 X
a = tf.constant(10.0, name = "a") # 기울기
b = tf.constant(4.45, name = "b") # 절편
Y = tf.constant(55.0, name = "y_data") # 정답

# name_scope
with tf.name_scope("Regress_model") as scope:
    model = (X * a) + b # y 예측치
    
with tf.name_scope("Model_error") as scope:
    model_err = tf.subtract(Y, model) # 부호 절대값
    
with tf.name_scope("Model_evaluation") as scope:
    square = tf.square(model_err)
    mse = tf.reduce_mean(square) # mse
    
with tf.Session() as sess:
    tf.summary.merge_all() # tensor 모으는 역
    writer = tf.summary.FileWriter("C:/ITWILL/6_Tensorflow/graph", sess.graph)
    writer.close()
    print("X =", sess.run(X))
    print("Y =", sess.run(Y))
    print("y pred =", sess.run(model))
    print("error =", sess.run(model_err))
    print("mse =", sess.run(mse))