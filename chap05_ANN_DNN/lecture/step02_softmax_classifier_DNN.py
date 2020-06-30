# -*- coding: utf-8 -*-
"""
step02_softmax_classifier_DNN

DNN model
 - hidden layer : Relu 활성함수 -> 지나치게 값이 작아지기 때문에
 - output layer : Softmax 활성함수
 - 2개 은닉층을 갖는 분류기
 - hidden1_nodes : 12개
 - hidden2_nodes : 6개
 - dataset : iris
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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

# 75 vs 25
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data) # 75 vs 25

# 2. X, Y변수 정의
X = tf.placeholder(dtype = tf.float32, shape = [None, 4]) # [관측치, 입력수]
Y = tf.placeholder(dtype = tf.float32, shape = [None, 3]) # [관측치, 출력수]

#########################
### DNN network
#########################

hidden1_nodes = 12
hidden2_nodes = 6

# hidden1 layer : 1층 : relu()
w1 = tf.Variable(tf.random_normal([4,hidden1_nodes])) # [input, output]
b1 = tf.Variable(tf.random_normal([hidden1_nodes]))   # [output]
hidden1_output = tf.nn.relu(tf.matmul(X, w1) + b1)

# hidden2 layer : 2층 : relu()
w2 = tf.Variable(tf.random_normal([hidden1_nodes,hidden2_nodes])) # [input, output]
b2 = tf.Variable(tf.random_normal([hidden2_nodes]))   # [output]
hidden2_output = tf.nn.relu(tf.matmul(hidden1_output, w2) + b2)

# output layer : 3층 : softmax()
w3 = tf.Variable(tf.random_normal([hidden2_nodes, 3])) # [input, output]
b3 = tf.Variable(tf.random_normal([3]))                # [output]

# 4. softmax 분류기 
# 1) 회귀방정식 : 예측치 
model = tf.matmul(hidden2_output, w3) + b3

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

bin(9).replace("0b", "")


lst = [1, 2, 3]
lst2 = list()    
max(lst2)
    
while True:
    print(lst)
    if len(lst) == 1:
        print("finis")
        break
    a = lst[-1]
    print(a)
    del lst[-1]
    b = lst[-1]
    print(b)
    lst2.append(a - b - 1)