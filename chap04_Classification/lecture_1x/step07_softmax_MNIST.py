# -*- coding: utf-8 -*-
"""
step07_softmax_MNIST

Softmax + MNIST
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. MNIST dataset load
minist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = minist.load_data()

x_train.shape # iamges(픽셀)  : (60000, 28, 28) [size, height, width]
y_train.shape # label(10진수) : (60000,)

# 첫번째 image 확인
plt.imshow(x_train[0]) # 5

x_train[0][10]
y_train[0] # 5

# images 전처리
# 1) images 정규화
x_train_nor, x_test_nor = x_train / 255.0, x_test / 255.0
x_train_nor[0][10]

x_train[0][10]; x_train_nor[0][10]

# 3차원 -> 2차원
x_train_nor = x_train_nor.reshape(-1, 784)
x_test_nor = x_test_nor.reshape(-1, 784)

x_train_nor.shape # (60000, 784)
x_test_nor.shape  # (10000, 784)

# labels 전처리
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

obj = OneHotEncoder(sparse = False)

y_train_one = obj.fit_transform(y_train)
y_test_one = obj.fit_transform(y_test)
y_train_one.shape # (60000, 10)
y_test_one.shape  # (10000, 10)

# 4. X,Y 변수 정의
X = tf.placeholder(dtype = tf.float32, shape = [None, 784]) # x_data
Y = tf.placeholder(dtype = tf.float32, shape = [None, 10])  # y_data

w = tf.Variable(tf.random_normal([784,10])) # [input, output]
b = tf.Variable(tf.random_normal([10]))     # [output]

# 5. softmax 알고리즘
# (1) model
model = tf.matmul(X, w) + b

# (2) softmax
softmax = tf.nn.softmax(model)

# (3) loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = model))

# (4) encoding -> decoding
y_pred = tf.argmax(softmax, 1)
y_true = tf.argmax(Y, 1)


train = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(100):
        _, loss_val = sess.run([train, loss], feed_dict = {X:x_train_nor,Y:y_train_one})
        if (step + 1) % 10 == 0:
            print(f"step = {step+ 1}, loss = {loss_val}")
    
    # model test
    feed_data2 = {X:x_test_nor, Y:y_test_one}
    y_pred_re = sess.run(y_pred, {X: x_test_nor})
    y_true_re = sess.run(y_true, {Y: y_test_one})
    print(y_true_re)
    print(y_pred_re)
    print(accuracy_score(y_true_re, y_pred_re))

