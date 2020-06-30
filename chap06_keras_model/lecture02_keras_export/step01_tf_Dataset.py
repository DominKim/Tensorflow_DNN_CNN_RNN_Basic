# -*- coding: utf-8 -*-
"""
lecture02_keras_export

step01_tf_Dataset.py

Dataset 클래스
 - dataset으로 부터 사용가능한 데이터를 메모리에 로딩 가능
 - batch size 지정
"""

import tensorflow as tf

from tensorflow.python.data import Dataset

# member 확인
dir(Dataset)
'''
batch()
from_tensor_slices()
shuffle()
'''

# 1. from_tensor_slices() : 입력 tensor로 부터 slice 생성
# ex) MNIST(60000, 28, 28) -> 60000개 image를 각각 1개식 slice

# 1) x,y변수 생성
x = tf.random.normal([5, 2])
y = tf.random.normal([5])

# 2) Dataset : 5개 slice
train_ds = Dataset.from_tensor_slices( (x,y) )
train_ds # ((2,), ()), types: (tf.float32, tf.float32)

# 5개 관측치 -> 5개 slices
for train_x, train_y in train_ds:
    print("x = {}, y = {}".format(train_x.numpy(), train_y.numpy()))
'''
x = [0.9365801  0.81943375], y = 0.18668098747730255
x = [-1.0868474  2.7385983], y = 0.4488779902458191
x = [-1.2210449  1.7996168], y = 1.1415643692016602
x = [ 1.0433679 -0.487757 ], y = -1.29880690574646
x = [ 1.7637053  -0.27344197], y = -1.576798915863037
'''

# 2. from_tensor_slices(x, y).shuffle(buffer size).batch
'''
shuffle(buffer size) : tensor 행단위 셔플링
   - buffer size : 전체 dataset에서 셔플링 size
batch : model에 1회 공급할 dataset size
ex) 60,000(mnist) -> shuffle(10000).batch(100)
    1번째 slice data : 10000개 셔플링 -> 100개씩 추출
    2번째 slice data :  -> 100개씩 추출
'''

# 1) x,y변수 생성
x = tf.random.normal([5, 2])
y = tf.random.normal([5])

# 2) Dataset : 5개 slice -> 3개 slice
train_ds2 = Dataset.from_tensor_slices( (x,y) ).shuffle(5).batch(2)
train_ds2 # ((None, 2), (None,)), types: (tf.float32, tf.float32)

# 3) 3 slice -> 1 slice
for train_x, train_y in train_ds2:
    print("x = {}, y = {}".format(train_x.numpy(), train_y.numpy()))
'''
x = [[-0.08522906  1.5093303 ]
 [ 0.9706549   0.08738968]], y = [ 0.30413634 -0.17981407]
x = [[-0.16985227  0.07214908]
 [-0.29674473 -0.40977556]], y = [0.61470354 0.33720043]
x = [[-0.45818874  1.5088389 ]], y = [-0.90355146]
'''

# 3. keras dataset 적용
from tensorflow.keras.datasets.cifar10 import load_data
import matplotlib.pyplot as plt

# 1. dataset load
(x_train, y_train), (x_val, y_val) = load_data()

x_train.shape # images : (50000, 32, 32, 3) - (size, h, w, c)
y_train.shape # (50000, 1)

plt.imshow(x_train[0])

y_train[0] # [6]

# train set batch size = 100 image
train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(100)

cnt = 0
for img_x, label_x in train_ds:
    print("image = {}, label = {}".format(img_x.shape, label_x.shape))
    cnt += 1
    
print("slice 개수 =", cnt) # slice 개수 = 500
# epochs = iter size(500) * btachsize(100)

# val set batch size = 100 image
test_ds = Dataset.from_tensor_slices( (x_val, y_val)).shuffle(10000).batch(100)

cnt = 0
for img_x, label_x in test_ds:
    print("image = {}, label = {}".format(img_x.shape, label_x.shape))
    cnt += 1
    
print("slice 개수 =", cnt) # slice 개수 = 100

'''
문) MNIST 데이터셋을 이용하여 train_ds, val_ds 생성하기
    train_ds : shuffle = 10,000, batch size = 32
    val_ds : batch size = 32
'''
from tensorflow.keras.datasets.mnist import load_data

# 1. dataset load
(x_train, y_train), (x_val, y_val) = load_data()
y_val

# 2. dataset 생성
train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
train_ds
val_ds = Dataset.from_tensor_slices((x_val, y_val)).batch(32)
val_ds

# 3. slices 확인
for x, y in train_ds:
    print(f"image = {x.shape}, label = {y.shape}")

for x, y in val_ds:
    print(f"image = {x.shape}, label = {y.shape}")




