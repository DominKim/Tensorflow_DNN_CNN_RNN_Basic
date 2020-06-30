# -*- coding: utf-8 -*-
"""
step05_Tfidf_sigmoid_DNN.py

 1. Tfidf 가중치 기법 - sparse matrix
 2. ham(0) / spam(1)
 3. Hyper parameters
    max features  = 4000  (input node)
    lr = 0.01
    epochs = 50
    batch size = 500
    iter size = 10
     -> 1epoch = 500 * 10 = 5,000
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import accuracy_score
import numpy as np


# file load allow_pickle = True(load 필수 입력)
x_train, x_test, y_train, y_test = np.load("C:/ITWILL/6_Tensorflow/data/spam_data.npy", allow_pickle = True)

print(x_train.shape) # (3901, 4000)
print(x_test.shape)  # (1673, 4000)
type(x_train) # numpy.ndarray
type(y_train) # list

# list -> numpy
y_train = np.array(y_train)
y_test = np.array(y_test)
type(y_train) # numpy.ndarray

# reshape
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Hyper parameters
max_features = 4000 # (input node)
lr = 0.01
epochs = 50
batch_size = 500
iter_size = 10

# X,Y 변수 정의
X = tf.placeholder(dtype = tf.float32, shape = [None, max_features])
Y = tf.placeholder(dtype = tf.float32, shape = [None, 1])

#########################
### DNN network
#########################

hidden1_nodes = 6
hidden2_nodes = 3

# hidden1 layer : 1층 : relu()
w1 = tf.Variable(tf.random_normal([max_features,hidden1_nodes])) # [input, output]
b1 = tf.Variable(tf.random_normal([hidden1_nodes]))   # [output]
hidden1_output = tf.nn.relu(tf.matmul(X, w1) + b1)

# hidden2 layer : 2층 : relu()
w2 = tf.Variable(tf.random_normal([hidden1_nodes,hidden2_nodes])) # [input, output]
b2 = tf.Variable(tf.random_normal([hidden2_nodes]))   # [output]
hidden2_output = tf.nn.relu(tf.matmul(hidden1_output, w2) + b2)

# output layer : 3층 : softmax()
w3 = tf.Variable(tf.random_normal([hidden2_nodes, 1])) # [input, output]
b3 = tf.Variable(tf.random_normal([1]))                # [output]
                                            
# 1) 회귀방정식 : 예측치 
model = tf.matmul(hidden2_output, w3) + b3

# sigmoid(예측치)
sigmoid = tf.sigmoid(model) 

# (2) loss function : Cross Entropy 이용 : -sum(Y * log(model)) 
'''
1차 방법
loss = -tf.reduce_mean(Y * tf.log(softmax) + (1 - Y) * tf.log(1 - softmax))
'''

# 2차 방법 : Sigmoid + CrossEntropy
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, 
                                                             logits = model))


# 3) optimizer : 오차 최소화(w, b update) 
train = tf.train.AdamOptimizer(lr).minimize(loss) # 오차 최소화

# 4) argmax() : encoding(2) -> decoding(10)

y_pred = tf.cast(sigmoid > 0.5, tf.float32) # y1:0.8,y2:0.1,y3:0.1

# 5. 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    
    # epochs = 50 -> 5000 * 50 = 250,000
    for epoch in range(epochs):
        loss_sum = 0
        for step in range(iter_size):
            idx = np.random.choice(a = x_test.shape[0], 
                                   size = batch_size, replace = False)
            _, loss_val = sess.run([train, loss], 
                                   feed_dict = {X:x_train[idx], Y:y_train[idx]})
            loss_sum += loss_val
        print(f"epoch = {epoch}, loss = {loss_sum / iter_size}")
    
    # mode result
    y_pred_re = sess.run(y_pred, feed_dict = {X:x_test}) # 예측치
    y_true_re = sess.run(Y, feed_dict = {Y:y_test}) # 정답
    acc = accuracy_score(y_true_re, y_pred_re)
    
    print("y_pred =", y_pred_re.reshape(-1))
    print("y_true =", y_true_re.reshape(-1))
    print("accuracy =", acc)
    























