# -*- coding: utf-8 -*-
"""
celeb image split(celeb_crop_5)
 1. image read
 2. x_test(50) vs x_train(51~200) split
 3. label : 0~4(celeb5) -> 10000 ~ 00001
 4. data set -> save(data 폴더)
"""

from PIL import Image # read/save
import os # dir 
from glob import glob # *.jpg

import numpy as np # array
from tensorflow.keras.utils import to_categorical

def image_data_split() :
    path = os.getcwd() # 현재 경로 
    #print(path)
    
    # x,y data save 
    x_train = []; y_train=[] # train image/label
    x_val = []; y_val = [] # val image/label
    
    dirs = os.listdir(path+'/train_celeb5/')
    print(dirs) # ['Bae_DooNa', 'Choi_MinSik', 'Cho_InSung', 'Cho_JinWoong', 'Cho_SeungWoo']
    
    label = 0
    for name in dirs : 
        
        n_split = 0 # test(50)/train(150) split
        
        # 1. image read
        for pic in glob(path+'/train_celeb5/'+name+'/*.jpg') :
            img = Image.open(pic) # image read
            img = np.array(img)   
            
            # 2. x_test(50) vs x_train(51~200) split
            if n_split < 50 : # test(50)
                x_val.append(img) # image 
                y_val.append([label]) # 정답 
            else : # train(150)
                x_train.append(img) 
                y_train.append([label])
            
            n_split += 1 
        
        # 3. label : 0~4(celeb5)
        label += 1 # (0~4)
    
    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)
            
x_train, y_train, x_val, y_val = image_data_split()
print(x_train.shape) # (740, 150, 150, 3)
print(y_train.shape) # (740,1)
print(x_val.shape) # (250, 150, 150, 3)
print(y_val.shape) # (250, 1)

print(x_train) # 0 ~ 255 -> 정규화, float32
print(y_train) # one-hot 

# data 정규화 

# 1. x_data : 정규화, float32
x_train = x_train / 255
x_val = x_val / 255

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
print(x_train)

# 2. y_data : one-hot 
y_train = to_categorical(y_train, 5)
y_val = to_categorical(y_val, 5)
print(y_train)
print(y_train.shape) # (740, 5)
print(y_val.shape) # (250, 5)

# image data save
image_xy = (x_train, y_train, x_val, y_val)
np.save('./create_file/image_train_test.npy', image_xy)






