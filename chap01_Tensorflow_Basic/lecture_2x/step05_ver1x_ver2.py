# -*- coding: utf-8 -*-
"""
step05_ver1x_ver2.py

ver1.x -> ver2.x

''' step08_variable_feed_csv.py -> ver2.0
 1. 즉시 실행
 2. 세션 대신 함수
 3. @tf.function 함수 장식
'''
"""

import tensorflow as tf
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


# 3단계. train / test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)

x_train.shape # (105, 4)
x_test.shape  # (45, 4)

# 4단계. session object : data 공급 -> 변수 
# 훈련용 data 공급
import numpy as np
X_val = tf.constant(np.array(x_data))
Y_val = tf.constant(np.array(y_data))
print(X_val.numpy())
print(Y_val.numpy())

# 평가용 data 공급
X_val2 = tf.constant(np.array(x_test))
Y_val2 = tf.constant(y_test)
print(X_val2.numpy())
print(Y_val2.numpy())
print(Y_val2.numpy().shape) # (45,)
print(type(Y_val2.numpy())) # <class 'numpy.ndarray'>

# numpy -> pandas 변경
X_df = pd.DataFrame(X_val2.numpy())
print(X_df.info())
print(X_df.mean(axis = 0))

Y_ser = pd.Series(Y_val2.numpy())
print(Y_ser.value_counts())


