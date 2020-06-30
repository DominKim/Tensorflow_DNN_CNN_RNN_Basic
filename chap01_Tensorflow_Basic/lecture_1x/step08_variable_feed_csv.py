# -*- coding: utf-8 -*-
"""
step08_variable_feed_csv.py
 - csv(pandas object) -> tensorflow variable
"""

import tensorflow.compat.v1 as tf # ver 2.x -> ver 1.x
tf.disable_v2_behavior() # ver2.x 사용

import pandas as pd # csv file read
from sklearn.model_selection import train_test_split # data split
path = "C:/ITWILL/6_Tensorflow/data"

iris = pd.read_csv(path + "/iris.csv")
iris.columns = iris.columns.str.replace(".", "_")
iris.info()

# 1단계. 공급 data 생성 : DataFrame
cols = list(iris.columns)
x_data = iris[cols[:4]]
y_data = iris[cols[-1]]

x_data.shape # (150, 4)
y_data.shape # (150,)

# 2단계. X, Y 변수 정의 : tensorflow변수 정의
X = tf.placeholder(dtype = tf.float32, shape = [None, 4]) # [?, 4]
Y = tf.placeholder(dtype = tf.string, shape = [None]) # [?] : 가변길이(1차원)

# 3단계. train / test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)

x_train.shape # (105, 4)
x_test.shape  # (45, 4)

# 4단계. session object : data 공급 -> 변수 
with tf.Session() as sess:
    # 훈련용 data 공급
    train_feed_data = {X : x_train, Y: y_train}
    X_val, Y_val = sess.run([X, Y], feed_dict = train_feed_data)
    print(X_val)
    print(Y_val)

    # 평가용 data 공급
    test_feed_data = {X : x_test, Y: y_test}
    X_val2, Y_val2 = sess.run([X, Y], feed_dict = test_feed_data)
    print(X_val2)
    print(Y_val2)
    print(Y_val2.shape) # (45,)
    print(type(Y_val2)) # <class 'numpy.ndarray'>

    # numpy -> pandas 변경
    X_df = pd.DataFrame(X_val2, columns = cols[:4])
    print(X_df.info())
    print(X_df.mean(axis = 0))
    
    Y_ser = pd.Series(Y_val2)
    print(Y_ser.value_counts())
    
