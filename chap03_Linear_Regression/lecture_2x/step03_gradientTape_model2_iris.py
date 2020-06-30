# -*- coding: utf-8 -*-
"""
step03_gradientTape_model2_iris.py

tf.GradientTape + regression model(iris)
 - x변수 : 2 ~ 4컬럼
 - y변수 : 1컬럼
 - model 최적화 알고리즘 : Adam
"""
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# 1. input/output 변수 정의 
iris = load_iris()
inputs = iris.data[:,1:]
outputs = iris.data[:,0]
inputs.shape # (150, 3) : X변수
outputs.shape # (150,)  : Y변수

x_train, x_test, y_train, y_test = train_test_split(
    inputs, outputs, test_size = 0.3, random_state = 123)

tf.random.set_seed(123) # W, B seed값
                   
# 2. model : Model 클래스
class Model(tf.keras.Model): # 자식클래스(부모클래스)
    def __init__(self): # 생성자
        super().__init__() # 부모생성자 호출
        self.W = tf.Variable(tf.random.normal([3,1])) # 기울기(가중치)
        self.B = tf.Variable(tf.random.normal([1])) # 절편
        
    def call(self, inputs): # 메서드 재정의, call : .call이 필요없이 생성자 객체 내에 인수입력 가능 
        # cast() : float64 -> float32
        return tf.matmul(tf.cast(inputs, tf.float32), self.W) + self.B # 회귀방정식(예측치)
        
    
# 3. 손실 함수 : 오차 반환
def loss(model, inputs, outputs):
  err = model(inputs) - outputs # 예측치 - 정답
  return tf.reduce_mean(tf.square(err)) # MSE

# 4. 미분계수(기울기) 계산  
def gradient(model, inputs, outputs):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, outputs) # 손실함수 호출  
        grad = tape.gradient(loss_value, [model.W, model.B]) 
        # 미분계수 -> 기울기와 절편 업데이트
    return grad # 업데이트 결과 반환

# 5. model 생성
model = Model() # 생성자

# 6. model 최적화
opt = tf.keras.optimizers.SGD(learning_rate = 0.01)

print("초기 손실값 : {:.6f}".format(loss(model, inputs, outputs)))
print("w : {}, b : {}".format(model.W.numpy(), model.B.numpy()))

# 7. 반복학습 : train
for step in range(500):
    grad = gradient(model, x_train, y_train) # 기울기 계산
    # 기울기 -> 최적화 객체 반영
    opt.apply_gradients(zip(grad, [model.W, model.B]))
    if (step + 1) % 20 == 0:
        print("step = {}, loss = {}".format(step + 1, 
                                            loss(model, x_train, y_train)))
    
# model 최적화
print("최종 손실값 : {:.6f}".format(loss(model, x_train, y_train)))
print("w : {}, b : {}".format(model.W.numpy(), model.B.numpy()))

# model test : test

y_pred = model.call(x_test)
# print(y_pred.numpy())


mse = mean_squared_error(y_test, y_pred)
print("mse =", mse)

r2 = r2_score(y_test,y_pred)
print("r2 =", mse)