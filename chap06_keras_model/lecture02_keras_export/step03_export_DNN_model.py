# -*- coding: utf-8 -*-
"""
step03_export_DNN_model.py

tensorflow 2.x 전문가용 DNN model 구축
 - tensorflow 2.0 저수준 API
 - Dataset 클래스 이용 : 공급 data
 - 순방향 step : 회귀방정식 연산 -> 예측치 -> loss
 - 역방향 step : 자동미분계수 계산 -> w, b update(model 최적화)
 - 손실함수, 최적화, 모델 평가 API
"""

import tensorflow as tf # ver 2.0
from tensorflow.python.data import Dataset # dataset 생성
from tensorflow.keras.layers import Dense, Flatten  # layer 추가
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras import datasets # mnist, cifar10

# 1. dataset load

mnist = datasets.mnist
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train.shape # (60000, 28, 28)
y_train.shape # (60000,)


# images 2d -> 1d
'''
x_train = x_train.reshape(-1, 28*28)
x_val = x_val.reshape(-1,28*28)
x_train.shape # (60000, 784)
x_val.shape   # (10000, 784)
'''
# iamges 정규화(0 ~ 1) -> 실수형
x_train = x_train / 255.
x_val = x_val / 255.
x_val[0]



# labels
y_train[0]
y_train # [5, 0, 4, ..., 5, 6, 8], dtype=uint8)

# 2. Dataset 생성
train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
train_ds # ((None, 784), (None,)), types: (tf.float64, tf.uint8) : None = btach size
val_ds = Dataset.from_tensor_slices((x_val, y_val)).batch(32)
val_ds   # ((None, 784), (None,)), types: (tf.float64, tf.uint8)

input_shape = (28, 28)

# 3. 순방향 step : 연산 -> (예측치 vs 관측치) -> loss
class Model(tf.keras.Model): # 자식클래스(부모클래스)
    def __init__(self): # 생성자
        super().__init__() # 부모생성자 호출
        # w, b 자동 초기화
        # DNN layer
        # 2d -> 1d
        self.d0 = Flatten(input_shape = input_shape) # flatten layer
        self.d1 = Dense(128, activation="relu") # hidden layer1
        self.d2 = Dense(64, activation="relu") # hidden layer2
        self.d3 = Dense(10, activation="softmax") # output layer
        
    def call(self, inputs): # inputs = images(32)
        # 회귀방정식 생략
        x = self.d0(inputs)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x) # 예측치(확률) 반환
    
# 4. loss function : 손실 함수 : 오차 반환
loss = losses.SparseCategoricalCrossentropy(from_logits = True)
# y_true(integer) vs y_pred(prop) : from_logits = True)

import numpy as np

# 손실이 작은 경우
y_true = np.array([0, 2]) # 정답 : 10진수
y_pred = np.array([[0.9, 0.02, 0.08], [0.1, 0.1, 0.8]]) # 예측치 : 확률

loss(y_true, y_pred) # 손실함수 : 0.6538635492324829

# 손실이 큰 경우
y_true = np.array([0, 1]) # 정답 : 10진수
y_pred = np.array([[0.9, 0.02, 0.08], [0.1, 0.1, 0.8]]) # 예측치 : 확률

loss(y_true, y_pred) # 손실함수 : 1.0038635730743408

# 5. model & optimizer
model = Model()
optimizer = optimizers.Adam()

# 6. model 평가 : loss, accuracy -> 1epoch 단위
train_loss = metrics.Mean() # loss mean
train_acc = metrics.SparseCategoricalAccuracy() # accuracy

val_loss = metrics.Mean() # loss mean
val_acc = metrics.SparseCategoricalAccuracy() # accuracy

# 7. 역방향 step : 자동 미분계수 -> W, b update
@tf.function # 연산속도 향상
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # 1) 순방향 : loss 계산
        preds = model(images) # model.call(images) : 예측치
        loss_value = loss(labels, preds) # 손실함수(y_true, y_pred)
        
        # 2) 역방향 : 손실값 -> w, b update
        grad = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        
        # 3) 1epoch -> loss, accuracy save
        train_loss(loss_value) # loss mean
        train_acc(labels, preds)

@tf.function # 연산속도 향상
def test_step(images, labels):
    # 1) 순방향 : loss 계산
    preds = model(images) # model.call(images) : 예측치
    loss_value = loss(labels, preds)
    
    # 2) 역방향 : 없음
    
    # 3) 1epoch -> loss, accuracy save
    val_loss(loss_value) # loss mean
    val_acc(labels, preds)
    
# 8. model training
epochs = 10

for epoch in range(epochs): # 10 epochs
    # next eopoch
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()
    # model train
    for images, labels in train_ds:
        train_step(images, labels)
    
    # model val
    for images, labels in val_ds:
        test_step(images, labels)

    form = "epoch = {}, Train loss ={:.6f}, Train Acc = {:.6f}, Val loss ={:.6f}, Val Acc = {:.6f}"
    print(form.format((epoch+1), 
                      train_loss.result(), 
                      train_acc.result(), 
                      val_loss.result(), 
                      val_acc.result()))












