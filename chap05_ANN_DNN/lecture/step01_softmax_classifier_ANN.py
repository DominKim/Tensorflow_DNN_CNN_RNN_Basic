# -*- coding: utf-8 -*-
"""
step01_softmax_classifier_ANN

ANN model
 - hidden layer : Relu 활성함수 -> 지나치게 값이 작아지기 때문에
 - output layer : Softmax 활성함수
 - 1개 은닉층을 갖는 분류기 w = shpae(?,5), b = shape (5)
 - node : 5개
 - dataset : iris
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import numpy as np

iris = load_iris()

# 1. x, y data 공급

# x변수 : 1 ~ 4 컬럼
x_data = iris.data
x_data.shape # (150, 4)

# y변수 : 5컬럼
y_data = iris.target

# reshape
y_data = y_data.reshape(-1,1)
y_data.shape # (150, 1)

'''
0 -> 1 0 0
1 -> 0 1 0
2 -> 0 0 1
'''


oht = OneHotEncoder(sparse = False)
y_data = oht.fit_transform(y_data)
y_data.shape # (150, 3)

# 2. X, Y변수 정의
X = tf.placeholder(dtype = tf.float32, shape = [None, 4]) # [관측치, 입력수]
Y = tf.placeholder(dtype = tf.float32, shape = [None, 3]) # [관측치, 출력수]

#########################
### ANN network
#########################

hidden_node = 5

# hidden layer
w1 = tf.Variable(tf.random_normal([4,hidden_node])) # [input, output]
b1 = tf.Variable(tf.random_normal([hidden_node]))   # [output]

# output layer
w2 = tf.Variable(tf.random_normal([hidden_node, 3])) # [input, output]
b2 = tf.Variable(tf.random_normal([3]))

# 4. softmax 분류기 
# 1) 회귀방정식 : 예측치 
hidden_output = tf.nn.relu(tf.matmul(X, w1) + b1) # 회귀모델(hidden_layer) -> 활성함수(Relu)

# output layer 결과
model = tf.matmul(hidden_output, w2) + b2

# softmax(예측치)
softmax = tf.nn.softmax(model) # 활성함수 적용(0 ~ 1) : y1 : 0.8, y2 : 0.1, y3 : 0.1

# (2) loss function : Cross Entropy 이용 : -sum(Y * log(model)) 
'''
1차 방법
loss = -tf.reduce_mean(Y * tf.log(softmax) + (1 - Y) * tf.log(1 - softmax))
'''

# 2차 방법 : Softmax + CrossEntropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels = Y, logits = model))


# 3) optimizer : 오차 최소화(w, b update) 
train = tf.train.AdamOptimizer(0.1).minimize(loss) # 오차 최소화

# 4) argmax() : encoding(2) -> decoding(10)

y_pred = tf.argmax(softmax, axis = 1) # y1:0.8,y2:0.1,y3:0.1
y_true = tf.argmax(Y, axis = 1)

# 5. 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    
    # 반복학습 : 500회
    for step in range(1000):
        _, loss_val = sess.run([train, loss], feed_dict = {X:x_data, Y: y_data})
        if (step + 1) % 50 == 0:
            print(f"step = {step + 1}, loss = {loss_val}")
    
    # mode result
    print(sess.run(softmax, feed_dict={X:x_data}))
    y_pred_re = sess.run(y_pred, feed_dict = {X:x_data}) # 예측치
    y_true_re = sess.run(y_true, feed_dict = {Y:y_data}) # 정답
    acc = accuracy_score(y_true_re, y_pred_re)
    
    print("y_pred =", y_pred_re)
    print("y_true =", y_true_re)
    print("accuracy =", acc)
    
    '''
    y_pred = [0 1 1 0 0 1]
    y_true = [0 1 2 0 0 2]
    accuracy = 0.6666666666666666
    '''
    import matplotlib.pyplot as plt
    
    plt.plot(y_pred_re, c = "r")
    plt.plot(y_true_re, c = "b")
