'''
문) breast_cancer 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하시오.
  조건1> keras layer
       L1 =  30 x 64
       L2 =  64 x 32
       L3 =  32 x 2
  조건2> optimizer = 'adam',
  조건3> loss = 'binary_crossentropy'
  조건4> metrics = 'accuracy'
  조건5> epochs = 300 
'''

import tensorflow as tf # ver2.x
from sklearn.datasets import load_breast_cancer # data set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, OneHotEncoder
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Dense # DNN layer 
from sklearn.metrics import accuracy_score

# 1. breast_cancer data load
cancer = load_breast_cancer()

x_data = cancer.data
y_data = cancer.target
print(x_data.shape) # (569, 30) : matrix
print(y_data.shape) # (569,) : vector

# x_data : 정규화 
x_data = minmax_scale(x_data) # 0~1

# y변수 one-hot-encoding 
obj = OneHotEncoder()
# reshape(2d) -> one hot encoding -> numpy
y_one_hot = obj.fit_transform(y_data.reshape([-1, 1])).toarray() 
y_one_hot
'''
[1, 00]
[0, 10]
'''
y_one_hot.shape # (569, 2)


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_one_hot, test_size = 0.3)


# 3. keras model
model = Sequential()

# 4. DNN model layer 구축 
model.add(Dense(64, input_shape = (30,), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(2, activation="sigmoid"))

# 5. model compile 
model.compile(optimizer = "adam",
              loss = "binary_crossentropy",
              metrics = ["accuracy"])
# 6. model trainint 
model.fit(x = x_train, y = y_train,
          epochs = 300,
          verbose = 1,
          validation_data = (x_val, y_val))

# 7. model evaluation 
model.evaluate(x = x_val, y= y_val)
# accuracy: 0.9649

y_true = y_val
y_pred = model.predict(x_val)
y_pred = tf.cast(y_pred > 0.5, tf.float32)
acc = accuracy_score(y_true, y_pred)
print("accuracy =", acc)