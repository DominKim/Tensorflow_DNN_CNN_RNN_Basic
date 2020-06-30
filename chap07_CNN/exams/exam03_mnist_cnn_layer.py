'''
exam_mnist_cnn_layer
 ppt-p.31 내용으로 CNN model를 설계하시오.
'''

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함

from tensorflow.keras.datasets.mnist import load_data # ver2.0 dataset
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np

# minst data read
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,) : 10진수 
print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000,) : 10진수 

# image data reshape : [s, h, w, c]
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape) # (60000, 28, 28, 1)

# x_data : int -> float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print(x_train[0]) # 0 ~ 255

# x_data : 정규화 
x_train /= 255 # x_train = x_train / 255
x_test /= 255
print(x_train[0])

# y_data : 10 -> 2(one-hot)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# hyper parameters
learning_rate = 0.001
epochs = 15
batch_size = 100
iter_szie = int(60000 / batch_size) # 600

# X, Y 변수 정의 
X_img = tf.placeholder(tf.float32, shape=[None, 28, 28, 1]) # (?, 784)
Y = tf.placeholder(tf.float32, shape=[None, 10]) # (?, 10)

# conv1
Filter1 = tf.Variable(tf.random_normal([5,5,1,32]))

conv1 = tf.nn.conv2d(X_img, Filter1, strides = [1,1,1,1], padding = "SAME")
relu1 = tf.nn.relu(conv1)
pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides = [1,2,2,1],  padding = "SAME")
print(pool1) # Tensor("MaxPool_33:0", shape=(?, 14, 14, 32), dtype=float32)

# conv2
Filter2 = tf.Variable(tf.random_normal([5,5,32,64]))

conv2 = tf.nn.conv2d(pool1, Filter2, strides = [1, 1, 1, 1], padding = "SAME")
relu2 = tf.nn.relu(conv2)
pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides = [1,2,2,1],  padding = "SAME")
pool2

n = 7 * 7 *64
flatten = tf.reshape(pool2, [-1, n])
flatten # <tf.Tensor 'Reshape_3:0' shape=(?, 3136) dtype=float32>


hidden_nodes = 128
# DNN layer
# Hyper parameter
lr = 0.01   # 학습률
epochs = 10 # 전체 dataset 재사용 횟수
batch_size = 100 # 1회 data 공급 횟수(mini batch)
iter_size  = 600 # 반복횟수

w1 = tf.random_normal([n, hidden_nodes])
b1 = tf.random_normal([hidden_nodes])
hidden_layer = tf.nn.relu(tf.matmul(flatten, w1) + b1)

w2 = tf.random_normal([hidden_nodes, 10])
b2 = tf.random_normal([10])

model = tf.matmul(hidden_layer, w2) + b2
softmax = tf.nn.softmax(model)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = model))

y_pred = tf.argmax(softmax, 1)
y_true = tf.argmax(Y, 1)

train = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        tot_loss = 0
        for _ in range(iter_size):
            idx = np.random.choice(x_train.shape[0], size = batch_size, replace=False)
            feed_data = {X_img:x_train[idx], Y:y_train[idx]}
            _, loss_val = sess.run([train, loss], feed_dict = feed_data)
            tot_loss += loss_val
        avg_loss = tot_loss / iter_size
        print(f"epoch = {epoch + 1}, loss = {avg_loss}")
        
    y_pred_re = sess.run(y_pred, feed_dict = {X_img:x_test})
    y_true_re = sess.run(y_true, feed_dict = {Y:y_test})
    print("y_pred =", y_pred_re)
    print("y_true =", y_true_re)
    print("acc =", accuracy_score(y_true_re,y_pred_re))
            