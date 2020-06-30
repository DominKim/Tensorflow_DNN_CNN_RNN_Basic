# -*- coding: utf-8 -*-
"""
CNN Mdole 실제 업무 data 적용 
 1. test set 생성/전처리 
 2. model load/test
 3. model 평가 : accuracy, confusion matrix
"""

from PIL import Image # read/save
import os # dir 
from glob import glob # *.jpg

import numpy as np # array
from tensorflow.keras.utils import to_categorical # one-hot
from tensorflow.keras.models import load_model

def image_data() :
    path = os.getcwd() # 현재 경로 
       
    # x, y data save     
    x_test = []; y_test = [] # test image/label
    
    dirs = os.listdir(path+'/test_celeb5/')
    print(dirs) # ['Bae_DooNa', 'Choi_MinSik', 'Cho_InSung', 'Cho_JinWoong', 'Cho_SeungWoo']
    
    label = 0
    for name in dirs :         
        
        for pic in glob(path+'/test_celeb5/'+name+'/*.jpg') :
            img = Image.open(pic) # image read
            img = np.array(img)    
            
            x_test.append(img) # image 
            y_test.append([label]) # 정답 
            
        label += 1 # (0~4)
    
    return np.array(x_test), np.array(y_test)

# 1. image data 생성              
x_test, y_test = image_data()
print(x_test.shape) # (250, 150, 150, 3)
print(y_test.shape) # (250, 1)


# 2. data 전처리 
# 1) x_data : 정규화, float32
x_test = x_test / 255
x_test = x_test.astype('float32')

# 2) y_data : one-hot 
y_test = to_categorical(y_test, 5)
print(y_test.shape) # (991, 5)


# 3. model test
# 1) model load
model = load_model("./create_file/celeb_CNN_model.h5")

# 2) model test
pred = model.predict(x_test) # 예측치 

# 3) accuracy 
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = np.argmax(pred, 1)
y_true = np.argmax(y_test, 1)

score = accuracy_score(y_true, y_pred)
print('accuracy = ', score)
# accuracy =  0.9747729566094854

con_mat = confusion_matrix(y_true, y_pred)

# 2. confusion matrix heatmap
import matplotlib.pyplot as plt
import seaborn as sn # heatmap - Accuracy Score
 
plt.figure(figsize=(6,6)) # chart size
sn.heatmap(con_mat, annot=True, fmt=".3f", linewidths=.5, square = True);# , cmap = 'Blues_r' : map 색상 
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 18)
plt.show()



