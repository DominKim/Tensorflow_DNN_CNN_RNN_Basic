'''
Full batch vs Mini batch
'''

import numpy as np # log()
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import minmax_scale # 정규화
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함
import time # 실행 시간 측정 

# Hyper parameter
learning_rate = 0.01 # 학습율 
iter_size = 100 # 학습횟수 
batch_size = 5000 # 1회 공급 data

# 1. dataset load
X, y = fetch_california_housing(return_X_y=True)

# x data 정규화 
print(X.shape) # (20640, 8)
x_data = minmax_scale(X) # 정규화 

# y data 편향 제거
y_data = np.log(y)  

# 2. train/test split
x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, random_state=123) # test_size = 0.25
print(x_train.shape) # (15480, 8)
print(x_test.shape) # (5160, 8)

# X, Y변수 : 공급형 
X = tf.placeholder(tf.float32, [None, 8]) # x변수:2차원
Y = tf.placeholder(tf.float32, [None]) # y변수 : 1차원   

# a,b변수 : 0 초기화 or 난수  
a = tf.Variable(tf.zeros([8, 1])) # a행, X열 - 수 일치 
b = tf.Variable(tf.zeros([1]))

# a와 X 행렬곱 
y_pred = tf.matmul(X, a) + b # a행, X열 - 수 일치 

# 비용함수(손실 함수) 작성 : 
cost = tf.reduce_mean(tf.square(y_pred - Y))

# 경사항강법 최적화 수행 : 0.1(0.01-속도 느림)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# 비용 최소화를 목적으로 한 학습
train = optimizer.minimize(cost)


# 세션 생성하고 초기화
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수 초기화    
    
    # full batch : full data 공급 
    feed_data = {X: x_train, Y: y_train} # full data
    
    # model 학습(train set) : 100번 수행   
    chktime=time.time()
    for step in range(iter_size): # 15000 * 100
        # full data 공급 
        # _, cost_val = sess.run([train, cost], feed_dict = feed_data)
        
        # mini batch : 1step당 3회 학습결과 반영 
        # image data는 주로 mini batch 사용 why? 용량이 크기 때문
        
        for i in range(3) : # 0~2[0~5000, 5000~10000, 10000~15000]
            start = i * batch_size # 0 > 5000 > 10000
            end = (start) + batch_size # 5000 >  10000 > 15000
            print(' start=', start, 'end=', end)
            # mini data 공급             
            feed_data = {X: x_train[start:end], Y: y_train[start:end]}
            _, cost_val = sess.run([train, cost], feed_dict = feed_data)
        
        # 1step 학습 후 cost 출력 
        print('step=', step+1, ' cost val=', cost_val)
    
    # 최적화 model 평가
    print('*** model 평가(test set) ***')
    feed_data2 = {X: x_test, Y: y_test} # test set - full data 
    y_pred_re = sess.run(y_pred, feed_dict = feed_data2) 
    y_true = sess.run(Y, feed_dict = feed_data2)
    print("MSE = %.5f" % mean_squared_error(y_true, y_pred_re))    
    print("MAE = %.5f" % mean_absolute_error(y_true, y_pred_re))
    
    # 실제값 vs 예측값 평균 비교 
    print('예측값 평균=', y_pred_re.mean()) 
    print('실제값 평균=', y_true.mean()) 
    
    print("="*30)
    chktime=time.time()-chktime
    print("실행시간 : ",chktime)
