# -*- coding: utf-8 -*-
"""
model overfitting solution
  - Dropout 적용
  - image 증식
"""
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Convolution layer
from tensorflow.keras.layers import Dense, Dropout, Flatten # Affine layer
import os

# dir setting
base_dir = "./cats_and_dogs"

train_dir = os.path.join(base_dir, 'train_dir')
validation_dir = os.path.join(base_dir, 'validation_dir')

# Hyper parameters
img_h = 150 # height
img_w = 150 # width
input_shape = (img_h, img_w, 3) 

# 1. CNN Model layer 
print('model create')
model = Sequential()

# Convolution layer1 : kernel[3,3,3,32]
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape = input_shape))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer2 : kernel[3,3,32,64]
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Convolution layer3 : kernel[5,5,64,128], maxpooling() 제외 
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# Flatten layer :4d -> 2d
model.add(Flatten()) 
# 2차 적용 : 드롭아웃 - 과적합 해결 
model.add(Dropout(0.5)) # fully connected 층 이전에 배치 

# Affine layer(Fully connected layer1) : [n, 256]
model.add(Dense(256, activation = 'relu'))

# Output layer(Fully connected layer2) : [256, 1]
model.add(Dense(1, activation = 'sigmoid'))

# model training set : Adam or RMSprop 
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy', # one hot encoding
              metrics = ['accuracy'])

# 2. image file preprocessing : 이미지 제너레이터 이용  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("image preprocessing")

# 1차 적용 
#train_data = ImageDataGenerator(rescale=1./255)

# 2차 적용 : image 증식 - 과적합 해결
train_data = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 40, # image 회전 각도 범위(+, - 범위)
        width_shift_range = 0.2, # image 수평 이동 범위
        height_shift_range = 0.2, # image 수직 이용 범위  
        shear_range = 0.2, # image 전단 각도 범위
        zoom_range=0.2, # image 확대 범위
        horizontal_flip=True,) # image 수평 뒤집기 범위 

# 검증 데이터에는 증식 적용 안함 
validation_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=35, 
        class_mode='binary') # binary label
# Found 2000 images belonging to 2 classes.

validation_generator = validation_data.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=35, # [수정] batch size 올림 
        class_mode='binary')
# Found 1000 images belonging to 2 classes.

# 3. model training : 배치 제너레이터 이용 모델 훈련 
model_fit = model.fit_generator(
          train_generator, 
          steps_per_epoch=100, 
          epochs=20, # 20 epochs()
          validation_data=validation_generator,
          validation_steps=50) 


# 4. model history graph
import matplotlib.pyplot as plt
 
print(model_fit.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

loss = model_fit.history['loss'] # train
acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss'] # validation
val_acc = model_fit.history['val_accuracy']

epochs = range(1, len(acc) + 1)

# acc vs val_acc   
plt.plot(epochs, acc, 'bo', label='train acc')
plt.plot(epochs, val_acc, 'r', label='val acc')
plt.title('Training vs validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuray')
plt.legend(loc='best')
plt.show()

# loss vs val_loss 
plt.plot(epochs, loss, 'bo', label='train loss')
plt.plot(epochs, val_loss, 'r', label='val loss')
plt.title('Training vs validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

