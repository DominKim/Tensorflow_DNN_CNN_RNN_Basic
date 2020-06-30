'''
문) image.jpg 이미지 파일을 대상으로 파랑색 우산 부분만 slice 하시오.
'''

import matplotlib.image as mp_image
import matplotlib.pyplot as plt
import tensorflow as tf
filename = "C:/ITWILL/6_Tensorflow/data/image.jpg"
input_image = mp_image.imread(filename)
plt.imshow(input_image)

input_image_slice = tf.slice(input_image, [100, 20, 0], [-1, 548, -1])
plt.imshow(input_image_slice)

input_image_slice.shape
# [332, 548, 3]