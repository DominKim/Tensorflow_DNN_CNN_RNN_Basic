# -*- coding: utf-8 -*-
"""
step01_gradientTape_sigmoid

GradientTape + Sigmoid
"""

import tensorflow as tf # ver2.x
tf.executing_eagerly()  # True

# 1. input/output 변수 정의 
# x변수 : [hours, video]
inputs = tf.Variable([[1., 2.], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]) # [6,2]
outputs = tf.Variable([[0.], [0], [0], [1], [1], [1]]) # [6, 1]



# 2. model : Model 클래스
class Model(tf.keras.Model): # 자식클래스(부모클래스)
    def __init__(self): # 생성자
        super().__init__() # 부모생성자 호출
        self.W = tf.Variable(tf.random.normal([2, 1])) # 기울기[입력,출력]
        self.B = tf.Variable(tf.random.normal([1])) # 절편[출력]
    def call(self, inputs): # 메서드 재정의, call : .call이 필요없이 생성자 객체 내에 인수입력 가능 
        return tf.matmul(inputs, self.W) + self.B # 회귀방정식(예측치)
        
# 3. 손실 함수 : 오차 반환
def loss(model, inputs, outputs):
  sigmoid = tf.sigmoid(model(inputs))
  loss = -tf.reduce_mean(outputs * tf.math.log(sigmoid) + (1-outputs) * tf.math.log(1-sigmoid))
  return loss # Cross Entropy

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
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)

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
sigmoid = tf.sigmoid(model.call(inputs)) # 0 ~ 1확률
pred = tf.cast(sigmoid > 0.5, tf.float32) # 관계식 -> 1 or 0

y_true = tf.squeeze(outputs) # 2차원 -> 1차원
y_pred = tf.squeeze(pred)

print("pred :", y_pred.numpy())
print("outputs :", y_true.numpy())
'''
pred : [0. 0. 1. 1. 1. 1.]
outputs : [0. 0. 0. 1. 1. 1.]
'''
 
 
 
 



 
















