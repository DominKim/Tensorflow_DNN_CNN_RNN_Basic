'''
문2) women.csv 데이터 파일을 이용하여 선형회귀모델  생성하시오.
     [조건1] x변수 : height,  y변수 : weight
     [조건2] learning_rate=0.1
     [조건3] 학습 횟수 100회
     [조건4] 학습과정 출력 - step, cost, a, b
'''
import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 

import pandas as pd
from sklearn.metrics import mean_squared_error

women = pd.read_csv('c:/ITWILL/6_Tensorflow/data/women.csv')
print(women.info())
print(women)

# 1. x,y data 생성 
x_data = women['height']
y_data = women['weight']

print(x_data.max()) # 72
print(y_data.max()) # 164

# 2. 정규화(0~1)
x_data = x_data / 72

y_data = y_data / 164

# 3. X,Y변수 정의 
X = tf.placeholder(dtype=tf.float32, shape=[None]) #x_data 공급 
Y = tf.placeholder(dtype=tf.float32, shape=[None]) #y_data 공급 

# a,b변수 정의 - 난수 이용 
a = tf.Variable(tf.random_uniform([1], 0.1, 1.0))
b = tf.Variable(tf.random_uniform([1], 0.1, 1.0))

# 4. model 생성 : y_pred, err, cost 식 작성 
y_pred = tf.multiply(X, a) + b
err = tf.subtract(Y, y_pred)
loss = tf.reduce_mean(tf.square(err))

optimazor = tf.train.GradientDescentOptimizer(0.1)
train = optimazor.minimize(loss)

init = tf.global_variables_initializer()
# 5. Session object 생성 : 반복 학습 
with tf.Session() as sess:
    sess.run(init)
    a_val, b_val = sess.run([a,b])
    feed_data = {X:x_data, Y:y_data}
    
    for i in range(100):
        _, loss_val = sess.run([train, loss], feed_dict = feed_data)
        a_up, b_up = sess.run([a, b])
        print("step = %d, loss = %5f, a = %5f, b = %5f" % (i+1, loss_val,a_up, b_up))
        
        


    
    
    
    
    
    
    
    
    
  
    