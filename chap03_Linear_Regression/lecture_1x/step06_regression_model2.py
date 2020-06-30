# -*- coding: utf-8 -*-
"""
step06_regression_model2.py

 y 변수 : 1컬럼
 x 변수 : 2 ~ 4 컬럼
 model 최적화 알고리즘 : GD -> Adam
"""

import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error # model 평가
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# 1. 공급 data 생성
iris = pd.read_csv("c:/ITWILL/6_Tensorflow/data/iris.csv")

x_data = np.array(iris.iloc[:, 1:4])
x_data.shape # (150, 3) : 2차원

y_data = np.array(iris.iloc[:, 0])
y_data.shape # (150,)   : 1차원

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)

# 2. X, Y 변수 정의 : 공급형 변수
X = tf.placeholder(dtype = tf.float32, shape = [None, 3])
Y = tf.placeholder(dtype = tf.float32, shape = [None])

# 3. a(w), b 변수 정의 : 난수 초기값
a = tf.Variable(tf.random_normal(shape = [3, 1])) # [입력수, 출력수]
b = tf.Variable(tf.random_normal(shape = [1])) # [출력수]

# 4. model 생성
model = tf.matmul(X, a) + b

loss = tf.reduce_mean(tf.square(Y - model))

optimazor = tf.train.AdamOptimizer(0.15)
train = optimazor.minimize(loss)

init = tf.global_variables_initializer()
# 5. model 학습 -> model 최적화(쵲거의 a, b update 됨) 
with tf.Session() as sess:
    sess.run(init)
    a_val, b_val = sess.run([a,b])
    print("최초 기울기 : {}, 절편 {}".format(a_val, b_val))
    feed_data = {X: x_train, Y: y_train}
    
    # 반복학습 100회
    for i in range(200):
        _, loss_val = sess.run([train, loss], feed_dict = feed_data)
        
        print("step = %d, loss = %f" % (i + 1, loss_val))
        
    # model 최적화    
    a_up, b_up = sess.run([a,b])
    print("수정된 기울기 : {}, 절편 {}".format(a_up, b_up))
    
    # 테스트용 공급 data
    feed_data_set = {X:x_test, Y:y_test}
    
    # Y(정답) vs model(예측치)
    y_true = sess.run(Y, feed_dict = {Y: y_test})
    y_pred = sess.run(model, feed_dict = feed_data_set)
    print(mean_squared_error(y_true, y_pred))

'''
0.48309776
0.90583146
0.50903505
'''    

