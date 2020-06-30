# -*- coding: utf-8 -*-
"""
croped image reshape(150x150)
"""

from glob import glob # (*.jpg)
from PIL import Image # image file read
from skimage import io # image save
import numpy as np
import os

fpath = os.getcwd() + "/croped_images"
print(fpath)

# croped image -> 150x150
def imageReshape() : 
    img_h = img_w = 150
    
    image_reshape = [] # image save 
    
    for file in glob(fpath + "/*.jpg") :
        img = Image.open(file) 
        
        # image 규격화 
        img = img.resize( (img_h, img_w) )
        # PIL -> numpy
        img_data = np.array(img)
        print(img_data.shape)
        
        image_reshape.append(img_data)
    
    return np.array(image_reshape) # list -> numpy
        
image_reshape = imageReshape()    

print(image_reshape.shape) # (3, 150, 150, 3) - (s, h, w, c)

import matplotlib.pyplot as plt
    
for i in range(image_reshape.shape[0]) :
    img = image_reshape[i]
    plt.imshow(img)
    plt.show()
    
    io.imsave(fpath+"/croped" + str(i+1) + "_reshape" + ".jpg", img)