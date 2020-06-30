# -*- coding: utf-8 -*-
"""
step07_image_transpose.py

 - image 변환
"""

import matplotlib.image as img  # image read
import matplotlib.pyplot as plt # image show
import tensorflow as tf

filename = "C:/ITWILL/6_Tensorflow/data/packt.jpeg"
input_image = img.imread(filename)

print('input dim =', input_image.ndim) # dimension
# input dim = 3
print('input shape =', input_image.shape) #shape
# input shape = (80, 144, 3)

# image 원본 출력 
plt.imshow(input_image)
plt.show() 

# image transpose : 축 변경[0,1,2]
img_tran = tf.transpose(a = input_image, perm = [1,0,2])
plt.imshow(img_tran)


print(img_tran.shape) # (144, 80, 3)
