'''
 image slice
'''

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

# image slice
img_slice = tf.slice(input_image, [15,0,0],[16,-1,-1])

print(img_slice.shape) 

# image slice 출력 
plt.imshow(img_slice)
plt.show()

