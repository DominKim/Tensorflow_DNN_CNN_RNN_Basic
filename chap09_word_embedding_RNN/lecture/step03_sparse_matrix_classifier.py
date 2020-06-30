# -*- coding: utf-8 -*-
"""
step03_sparse_matrix_classifier.py

Sparse matrix(Tfidf) + DNN model

1. csv file load
2. label + texts
3. texts 전처리 : 텍스트 벡터화
4. 희소행렬(sparse matrix)
5. DNN model 생성
"""

import pandas as pd
import numpy as np
import string
from tensorflow.keras.preprocessing.text import Tokenizer # token 생성
from tensorflow.keras.preprocessing.sequence import pad_sequences # padding
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# 1. file load
temp_spam = pd.read_csv("C:/ITWILL/6_Tensorflow/data/temp_spam_data2.csv",
                        header = None, encoding = "utf-8")

temp_spam.info()
'''
 0   0       5574 non-null   object : label(ham or spam)
 1   1       5574 non-null   object : texts(영문장)
'''

# 변수 선택
label = temp_spam[0]
texts = temp_spam[1]
len(label) # 5574

# 2. data 전처리

# target dummy('spam'=1, 'ham'=0)
target = [1 if x=='spam' else 0 for x in label]
print('target :', target)
target = np.array(target)

# texts 전처리
def text_prepro(texts):
    # Lower case
    texts = [x.lower() for x in texts]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # Remove numbers
    texts = [''.join(c for c in x if c not in string.digits) for x in texts]
    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    return texts

texts = text_prepro(texts)
texts[0]

# 3. 토큰화 : texts -> token
tokenizer = Tokenizer(num_words = 4000) # 약 8000 -> 4000개 단어
tokenizer.fit_on_texts(texts)
token = tokenizer.word_index
print("전체 단어 수 =", len(token)) # 전체 단어 수 = 8629

# 4. 희소행렬
x_data = tokenizer.texts_to_matrix(texts, mode = "tfidf")
x_data.shape # (5574, 4000) : (docs, terms)

# 5. dataset split
x_train, x_val, y_train, y_val = train_test_split(x_data, target, test_size = 0.3)


# 6. DNN layer
input_shape = (4000,)

model = Sequential()

model.add(Dense(64, input_shape = input_shape, activation = "relu")) # 1층

model.add(Dense(32, activation="relu")) # 2층

model.add(Dense(1, activation="sigmoid"))

model.summary()

# 7. compile / training
model.compile(optimizer = "Adam", 
              loss = "binary_crossentropy",
              metrics = ["accuracy"])

model.fit(x_train, y_train,
          epochs = 5,
          batch_size = 512,
          verbose=1,
          validation_data=(x_val, y_val))

loss, score = model.evaluate(x_val, y_val)
print("loss = ", loss, "\n", "accuracy =", score)

y_pred = model.predict(x_val)
y_pred = tf.cast(y_pred > 0.5, tf.float32)
result = ["정답" if y_pred[i] == y_val[i] else "틀림" for i in range(len(y_pred))]
result










