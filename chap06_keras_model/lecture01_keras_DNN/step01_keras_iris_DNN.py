# -*- coding: utf-8 -*-
"""
step01_keras_iris_DNN

 - Tensroflow2.x Keras + iris
 - Keras : DNN model 생성을 위한 고수준 API
 - Y변수 : one hot encoding
    loss = "categorical_crossentropy"
    metrics=["accuracy"]
"""

import tensorflow as tf # ver2.x
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale    # x변수 전처리
from tensorflow.keras.utils import to_categorical # y변수 전처리
from tensorflow.keras import Sequential           # model 생성
from tensorflow.keras.layers import Dense         # layer 생성
from tensorflow.keras.models import load_model    # model load
from sklearn.metrics import accuracy_score

# 1. x, y data 공급
iris = load_iris()

# x변수 : 1 ~ 4 컬럼
x_data = iris.data
x_data.shape # (150, 4)

x_data = minmax_scale(x_data) # x변수 전처리 : 정규화

# y변수 : 5컬럼
y_data = iris.target

# reshape
# y_data = y_data.reshape(-1,1)
# y_data.shape # (150, 1)

# [수정]
y_data = to_categorical(y_data) # y변수 전처리 : one hot encoding
y_data.shape # (150, 3)

# 75 vs 25
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data) # 75 vs 25
print(x_train.shape) # (112, 4)
print(x_val.shape)   # (38, 4)

# 2. keras model 생성
model = Sequential()
model # object info

# 3. model layer
'''
model.add(Dense(node수, input_shape, activation)) # hidden layer1
model.add(Dense(node수, activation)) # hidden layer2 ~ n
'''
# hidden layer1 = [4, 12]
model.add(Dense(12, input_shape=(4,), activation="relu")) # 1층
# hidden layer2 = [12, 6]
model.add(Dense(6, activation="relu"))                    # 2층
# output layer = [6, 3]
model.add(Dense(3, activation="softmax"))                 # 3층(output layer)

# 4. model complie
model.compile(optimizer ="adam", # 최적화 알고리즘(lr 생략)
              loss = "categorical_crossentropy", # 손실 전제 조건 : y one_hot_encoding
              metrics = ["accuracy"]) # 평가 방법

# layer 확인
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 12)                60=w(4*12)+b(12)
_________________________________________________________________
dense_4 (Dense)              (None, 6)                 78=w(12*6)+b(6)        
_________________________________________________________________
dense_5 (Dense)              (None, 3)                 21=w(6*3)+b(3)
=================================================================
'''

# 5. model training : train(112) vs val(38)
model.fit(x=x_train, y=y_train, # 학습용
          epochs = 300,
          verbose = 1,
          validation_data=(x_val, y_val)) # 평가용

# 6. model evaluation : 모델 검증
model.evaluate(x = x_val, y = y_val) # accuracy: 0.9737


# 7. model save & load
model.save("keras_model_iris.h5")
print("model saved")

new_model = load_model("keras_model_iris.h5")
print(new_model)

# 8. model test : new dataset
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
y_pred = new_model.predict(x_test) # 예측치
y_true = y_test # 관측치

y_pred = tf.argmax(y_pred, 1)
y_true = tf.argmax(y_true, 1)

acc = accuracy_score(y_true, y_pred)
print("accuracy =", acc)
# accuracy = 0.9473684210526315









