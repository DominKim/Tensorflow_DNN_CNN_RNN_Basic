'''
문) 다음과 같이 Convolution layer와 Max Pooling layer를 정의하고, 실행하시오.
  <조건1> input image : volcano.jpg 파일 대상    
  <조건2> Convolution layer 정의 
    -> Filter : 6x6
    -> featuremap : 16개
    -> strides= 1x1, padding='SAME'  
  <조건3> Max Pooling layer 정의 
    -> ksize= 3x3, strides= 2x2, padding='SAME' 
'''

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../images/volcano.jpg') # 이미지 읽어오기
plt.imshow(img)
plt.show()
print(img.shape) # (450, 720, 3)

img = img.astype(np.float32)
img = img.reshape(-1,405,720,3)

img_re = img / 255


# 
Filter = tf.Variable(tf.random.normal([6,6,3,16]))

cv_2d = tf.nn.conv2d(img_re, Filter, strides = [1,1,1,1], padding = "SAME")
print(cv_2d) # Tensor("Conv2D_25:0", shape=(1, 405, 720, 16), dtype=float32)

poll = tf.nn.max_pool(cv_2d, ksize = [1,3,3,1], strides=[1,2,2,1], padding = "SAME")
print(poll) # Tensor("MaxPool_17:0", shape=(1, 203, 360, 16), dtype=float32)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    cv = sess.run(cv_2d)
    cv = np.swapaxes(cv, 0, 3)
    print(cv.shape)
    for i, img in enumerate(cv):
        plt.subplot(1, 16, i + 1)
        plt.imshow(img.reshape(405,720))
    plt.show()
        
    # 폴링 연산
    poll_re = sess.run(poll)
    poll_re = np.swapaxes(poll_re, 0, 3)
    for i, img in enumerate(poll_re):
        plt.subplot(1, 16, i + 1)
        plt.imshow(img.reshape(203,360))
    plt.show()