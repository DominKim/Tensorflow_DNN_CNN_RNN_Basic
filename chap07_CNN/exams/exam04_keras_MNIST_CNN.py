'''
문) 다음과 같은 조건으로 keras CNN model layer를 작성하시오.

1. Convolution1
    1) 합성곱층 
      - filters=64, kernel_size=5x5, padding='same'  
    2) 풀링층(max) 
     - pool_size= 2x2, strides= 2x2, padding='same'

2. Convolution2
    1) 합성곱층 
      - filters=128, kernel_size=5x5, padding='same'
    2) 풀링층
     - pool_size= 2x2, strides= 2x2, padding='same'
    
3. Flatten layer 

4. Affine layer(Fully connected)
    - 256 node, activation = 'relu'
    - Dropout : 0.5%

5. Output layer(Fully connected)
    - 10 node, activation = 'softmax'
'''

from tensorflow.keras.datasets.mnist import load_data # ver2.0 dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation # Convolution layer
from tensorflow.keras.layers import Dense, Dropout, Flatten # Affine layer

# minst data read
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,) : 10진수 
print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000,) : 10진수 

# image data reshape : [s, h, w, c]
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape) # (60000, 28, 28, 1)
print(x_train[0]) # 0 ~ 255

# x_data : int -> float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# x_data : 정규화 
x_train /= 255 # x_train = x_train / 255
x_test /= 255
print(x_train[0])

# y_data : 10 -> 2(one-hot)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 1. CNN Model layer
model = Sequential()

model.add(Conv2D(64, kernel_size = (5,5), activation="relu"))
model.add(MaxPool2D(pool_size = (5,5), strides = (2,2)))

model.add(Conv2D(128, kernel_size = (5,5), activation="relu"))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(10, activation="softmax"))


# 2. model 생성 
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy', # one hot encoding
              metrics = ['accuracy'])

# 3. model train
model_fit = model.fit(x=x_train, y=y_train, 
                      batch_size=100, 
                      verbose=1,
                      epochs=3)

# 4. model test
score = model.evaluate(x=x_test, y=y_test)
print('loss =', score[0])
print('accuracy =', score[1])

