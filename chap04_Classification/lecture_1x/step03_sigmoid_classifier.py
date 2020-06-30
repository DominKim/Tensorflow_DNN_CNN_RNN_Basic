# -*- coding: utf-8 -*-
"""
step03_sigmoid_classifier.py
 - 활성함수(activation function) : sigmoid
 - 손실함수(loss function) : cross entropy
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.metrics import accuracy_score # model 평가

# 1. x, y 공급 data 
# x변수 : [hours, video]
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] # [6,2]

# y변수 : binary data (fail or pass)
y_data = [[0], [0], [0], [1], [1], [1]] # [6, 1]

# 2. X, Y변수 정의
X = tf.placeholder(dtype = tf.float32, shape = [None, 2]) # [관측치, 입력수]
Y = tf.placeholder(dtype = tf.float32, shape = [None, 1]) # [관측치, 출력수]

# 3. w,b 변수 정의
w = tf.Variable(tf.random_normal([2, 1])) # [입력수, 출력수]
b = tf.Variable(tf.random_normal([1])) # [출력수]

# 4. sigmoid 분류기
# (1) model : 예측치
model = tf.matmul(X, w) + b # 회귀방정식
sigmoid = tf.sigmoid(model) # 활성함수 적용(0 ~ 1 확률)

# (2) loss function : Entropy 수식 = -sum(Y * log(model)) 
# loss = -tf.reduce_mean(Y * tf.log(sigmoid) + (1-Y) * tf.log(1-sigmoid))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = model))
# Y * tf.log(sigmoid) Y = 1인 경우 손실 값 계산
# (1-Y) * tf.log(1-sigmoid) Y = 0인 경우 손실 값 계산


# (3) optimizer 
'''
opt = tf.train.GradientDescentOptimizer(0.1)
train = opt.minimize(loss)
'''
train = tf.train.AdamOptimizer(0.1).minimize(loss) # 최적화 객체
# (4) cut-off : 0.5
cut_off = tf.cast(sigmoid > 0.5, tf.float32) # T/F -> 1.0/0.0

# 5. model training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # w, b 초기화
    
    feed_data = {X:x_data, Y:y_data} # 공급 data
    for step in range(500):
        _,loss_val = sess.run([train, loss], feed_dict = feed_data)
        if (step + 1) % 50 == 0:
            print(f"step = {step + 1}, loss = {loss_val}")
    # model 최적화
    y_true = sess.run(Y, feed_dict = {Y:y_data}) # 
    y_pred = sess.run(cut_off, feed_dict = {X:x_data})
    
    acc = accuracy_score(y_true, y_pred)
    print("accuracy =", acc)
    
    print("y_true :", y_true)
    print("y_pred :", y_pred)
    

    
    
