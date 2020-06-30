# -*- coding: utf-8 -*-
"""
step02_greadientTape_moel.py

tf.GradientTape + regression model
 -> 미분계수 자동 계산 -> model 최적화(최적의 기울기와 절편 update)
"""
import tensorflow as tf
tf.executing_eagerly()

# 1. input/output 변수 정의 
inputs = tf.Variable([1.0, 2.0, 3.0]) # x변수 
outputs = tf.Variable([2.0, 4.0, 6.0]) # y변수 : 1.25 > 1.9 > 2.8


# 2. model : Model 클래스
class Model(tf.keras.Model): # 자식클래스(부모클래스)
    def __init__(self): # 생성자
        super().__init__() # 부모생성자 호출
        self.W = tf.Variable(tf.random.normal([1])) # 기울기(가중치)
        self.B = tf.Variable(tf.random.normal([1])) # 절편
        
    def call(self, inputs): # 메서드 재정의, call : .call이 필요없이 생성자 객체 내에 인수입력 가능 
        return inputs * self.W + self.B # 회귀방정식(예측치)
        
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
'''
mse = loss(model, inputs, outputs)
print("mse =", mse.numpy()) # mse = 27.385439

grad = gradient(model, inputs, outputs)
print("grad =", grad)
'''

# 6. model 최적화
opt = tf.keras.optimizers.SGD(learning_rate = 0.01)

print("초기 손실값 : {:.6f}".format(loss(model, inputs, outputs)))
print("w : {}, b : {}".format(model.W.numpy(), model.B.numpy()))

# 7. 반복학습
for step in range(300):
    grad = gradient(model, inputs, outputs) # 기울기 계산
    # 기울기 -> 최적화 객체 반영
    opt.apply_gradients(zip(grad, [model.W, model.B]))
    if (step + 1) % 20 == 0:
        print("step = {}, loss = {}".format(step + 1, 
                                            loss(model, inputs, outputs)))
    
# model 최적화
print("최종 손실값 : {:.6f}".format(loss(model, inputs, outputs)))
print("w : {}, b : {}".format(model.W.numpy(), model.B.numpy()))

# model test
y_pred = model(2.5) # x = 2.5
print("y_pred =", y_pred.numpy()) # y_pred = [5.0071735]

'''
초기 손실값 : 0.832718
w : [1.7111176], b : [-0.30375817]


최종 손실값 : 0.000742
w : [2.0316322], b : [-0.07190713]
''' 
 

 
 
 
 

 