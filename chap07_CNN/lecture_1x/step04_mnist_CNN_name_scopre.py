# -*- coding: utf-8 -*-
"""
step04_mnist_CNN_name_scopre.py

- MNIST + CNN + name Scope + tensorboard

 0. input layer : image(?x28x28 -> ?x28x28x1) -> ?(-1)
 1. Conv layer1(Conv -> relu -> Pool)
 2. Conv layer2(Conv -> relu -> Pool)
 3. Flatten layer : 3D[size, height, width, color] -> 1D[s, n] : n = h*w*c
 4. DNN hidden1 layer : [s, n] * [n, node]
 5. DNN output layer :  [node, 10] * [node, 10]
"""

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함

from tensorflow.keras.datasets.mnist import load_data # dataset load
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# tensorboard 초기화
tf.reset_default_graph()

###########################
## 0. input layer
###########################

# 1. image read 
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,) : 10진수 

# 2. 실수형 변환 : int -> float32
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')

# 3. 정규화 
x_train = x_train / 255 # x_train = x_train / 255
x_test = x_test / 255

# 4. input image reshape  
x_train = x_train.reshape(-1,28,28,1)  # (size, h, w, color)
x_test = x_test.reshape(-1,28,28,1)    

# 5. y_data 전처리 : one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train.shape # (60000, 10)

# 6. X, Y변수 정의
X_img = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape = [None, 10])

# 합성곱 계층 함수
def conv2d_func(Img, Filter):
    return tf.nn.conv2d(input = Img,filter = Filter, strides=[1,1,1,1], padding='SAME')
    
# 폴링 계층 함수
def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1,2,1,1], strides= [1,2,2,1], padding = "SAME")

###########################################
## 1. Conv layer1(Conv -> relue -> Poll)
###########################################
with tf.name_scope("Convolution1") as scope:
    
    Filter1 = tf.Variable(tf.random_normal([3,3,1,32]))
    
    conv2 = conv2d_func(X_img, Filter1)
    L1 = tf.nn.relu(conv2) # 정규화 : 0 ~ x
    L1_out = max_pool(L1)
    print(L1_out) # Tensor("MaxPool_25:0", shape=(?, 14, 14, 32), dtype=float32)

###########################################
## 2. Conv layer1(Conv -> relue -> Poll)
###########################################
with tf.name_scope("Convolution2") as scope:
    Filter2 = tf.Variable(tf.random_normal([3,3,32,64]))
    
    conv2_l2 = conv2d_func(L1_out, Filter2)
    L2 = tf.nn.relu(conv2_l2) # 정규화 : 0 ~ x
    L2_out = max_pool(L2)
    print(L2_out) # Tensor("MaxPool_26:0", shape=(?, 7, 7, 64), dtype=float32)

#################################
## 3. Flatten layer (3d -> 1d)
#################################
with tf.name_scope("Flatten") as scope:
    n = 7 * 7 * 64 # 3d -> 1d
    L2_Flat  = tf.reshape(L2_out, [-1, n])
    print(L2_Flat) # Tensor("Reshape:0", shape=(?, 3136), dtype=float32)

# DNN layer
# Hyper parameter
lr = 0.01   # 학습률
epochs = 10 # 전체 dataset 재사용 횟수
batch_size = 100 # 1회 data 공급 횟수(mini batch)
iter_size  = 600 # 반복횟수

####################################
## DNN Network
####################################

with tf.name_scope("DNN_hidden_layer") as scope:
    hidden_nodes = 128
    
    # hidden layer : 1층 : relu()
    w1 = tf.Variable(tf.random_normal([n, hidden_nodes]), name = "w1") # [input, output]
    b1 = tf.Variable(tf.random_normal([hidden_nodes]), name = "b1")     # [output]
    hidden_output = tf.nn.relu(tf.matmul(L2_Flat, w1) + b1)

with tf.name_scope("DNN_output_layer") as scope:
    # output layer : 3층 : softmax()
    w2 = tf.Variable(tf.random_normal([hidden_nodes, 10]), name = "w2") # [input, output]
    b2 = tf.Variable(tf.random_normal([10]), name = "b2")               # [output]

    # 5. softmax 알고리즘
    # (1) model
    model = tf.matmul(hidden_output, w2) + b2
    
    # (2) softmax
    softmax = tf.nn.softmax(model)

with tf.name_scope("lossFunction") as scope:
    # (3) loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = model), name = "loss")

with tf.name_scope("optimizer") as scope:
    # (4) optimizer
    train = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.name_scope("Prediction") as scope:
    # (5) encoding -> decoding
    y_pred = tf.argmax(softmax, 1)
    y_true = tf.argmax(Y, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    tf.summary.merge_all() # tensor 모으는 역
    writer = tf.summary.FileWriter("C:/ITWILL/6_Tensorflow/graph", sess.graph)
    print("tensorboard 시각화 완료")
    writer.close()
    
    feed_data = {X_img:x_train, Y:y_train}
    
    # epochs = 10
    for epoch in range(epochs): # 1세대
        tot_loss = 0
        
        # 1epoch = 100 * 600
        for step in range(iter_size) : # 600번 반복 학습
            idx = np.random.choice(a = y_train.shape[0], 
                                   size = batch_size, replace = False)
            # Mini batch dataset
            feed_data = {X_img: x_train[idx], Y:y_train[idx]}
            _, loss_val = sess.run([train, loss], feed_dict = feed_data)
            
            tot_loss += loss_val
            
        # 1epoch 종료
        avg_loss = tot_loss / iter_size
        print(f"epoch ={epoch+1}, loss = {avg_loss}")
        
        
    # model 최적화 : test
    feed_data2 = {X_img:x_test, Y:y_test}
    y_pred_re = sess.run(y_pred, feed_dict = feed_data2)
    y_true_re = sess.run(y_true, feed_dict = feed_data2)
    print(y_true_re)
    print(y_pred_re)
    print(accuracy_score(y_true_re, y_pred_re))
