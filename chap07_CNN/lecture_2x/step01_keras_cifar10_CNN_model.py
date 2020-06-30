# -*- coding: utf-8 -*-
"""
step01_keras_cifar10_CNN_model.py

- Keras CNN model + cifar10

1. image dataset load
2. image 전처리 : 실수형, 정규화, one-hot encoding
3. Keras Model
4. Model 평가
5. Model history
"""

import tensorflow as tf # ver2.x
from tensorflow.keras import Sequential                     # model
from tensorflow.keras.datasets.cifar10 import load_data     # datase
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D    # Conv
from tensorflow.keras.layers import Dense, Flatten, Dropout # DNN
import time

start_time = time.time()

# 1. dataset load
(x_train, y_train), (x_val, y_val) = load_data()
x_train.shape # images : (50000, 32, 32, 3)
y_train.shape # labels : (50000, 1)

# 2. image 전처리 : 실수형 -> 정규화
x_train[0] # 0 ~ 255 : 정수형
x_train = x_train.astype("float32")
x_val = x_val.astype("float32")

# 정규화
x_train = x_train / 255
x_val = x_val / 255
x_train[0] # 전처리 확인

# label 전처리 : one-hot
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# 2. keras model layer
model = Sequential()

input_shape = (32, 32, 3)

# conv layer1 : 1층 [5,5,3,32] : kernel_size -> Filter
model.add(Conv2D(32, kernel_size=(5,5), input_shape = input_shape,
                 activation = "relu"))
model.add(MaxPooling2D(pool_size =(3,3), strides=(2,2)))
model.add(Dropout(0.2)) # 16x16

# conv layer2 : 2층 [5,5,32,64]
model.add(Conv2D(64, kernel_size=(5,5), activation = "relu"))
model.add(MaxPooling2D(pool_size =(3,3), strides=(2,2)))
model.add(Dropout(0.2)) # 8x8

# Flatten : 3D -> 1D
model.add(Flatten())

# DNN hidden layer : 3층
model.add(Dense(64, activation="relu"))

# DNN output layer : 4층
model.add(Dense(10, activation="softmax"))

# 3. model compile : 학습과정 설정(다항 분류기)
model.compile(optimizer = "adam",                # 최적화 알고리즘(lr 생략)
              loss = "categorical_crossentropy", # 손실
              metrics = ["accuracy"])            # 평가 방법

# layer 확인
model.summary()


# 4. model training : train(112) vs val(38)
model_fit = model.fit(x=x_train, y=y_train, # 학습용
          batch_size = 100, # 1회 공급 data size
          epochs = 10, # image 재학습 size
          verbose = 1,
          validation_data=(x_val, y_val)) # 평가용

# 5. model evaluation : 모델 검증
loss, acc = model.evaluate(x = x_val, y = y_val) # accuracy: 0.9737
print("loss ={}, accuracy = {}".format(loss, acc))

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# 6. model history
print(model_fit.history.keys())

# train vs val accuracy
train_loss = model_fit.history["loss"]
train_acc = model_fit.history["accuracy"]
val_loss = model_fit.history["val_loss"]
val_acc = model_fit.history["val_accuracy"]


import matplotlib.pyplot as plt

# train vs val loss
plt.plot(train_loss, c = "y", label = "train loss")
plt.plot(val_loss, c = "r", label = "val loss")
plt.legend(loc = "best")
plt.xlabel("epochs")
plt.show()

# train vs val accuracy
plt.plot(train_acc, c = "y", label = "train acc")
plt.plot(val_acc, c = "r", label = "val acc")
plt.legend(loc = "best")
plt.xlabel("epochs")
plt.show()

# 7. model test(new dataset)
from sklearn.metrics import classification_report # f1 score 
import numpy as np

idx = np.random.choice(a = x_val.shape[0], size = 100, replace=False)
x_test = x_val[idx] # new dataset images
y_test = y_val[idx] # new dataset labels

y_pred = model.predict(x_test) # 예측치 : 확률
y_true = y_test # 정답 : one-hot

# integer 변환
y_true = np.argmax(y_true, 1)
y_pred = np.argmax(y_pred, 1)

report = classification_report(y_true, y_pred)
print(report)
'''
              precision    recall  f1-score   support

           0       0.69      0.75      0.72        12
           1       0.62      1.00      0.76         8
           2       0.62      0.57      0.59        14
           3       0.46      0.60      0.52        10
           4       0.80      0.62      0.70        13
           5       0.40      0.33      0.36         6
           6       0.71      0.71      0.71         7
           7       0.56      0.62      0.59         8
           8       0.75      0.82      0.78        11
           9       1.00      0.45      0.62        11

    accuracy                           0.65       100
   macro avg       0.66      0.65      0.64       100
weighted avg       0.68      0.65      0.65       100
'''
# 성공여부
for i in range(100):
    if y_true[i] == y_pred[i]:
        print("success :", labels[y_true[i]])
    else:
        print("fail : real({}) -> pred({})".format(labels[y_true[i]], labels[y_pred[i]]))

end_time = time.time() - start_time
print("소요시간 :", end_time)