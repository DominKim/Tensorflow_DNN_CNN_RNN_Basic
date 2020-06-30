# -*- coding: utf-8 -*-
"""
step02_features_extract.py

1. 텍스트에서 특징(features) 추출 방법
 - 지도학습에서 text 대상 희소행렬(sparse matrix)
 - 방법 : tf, idf 방법
 
2. num_words(max features)
 - 단어의 차수를 지정하는 속성
 ex) num_words = 500 : 전체 단어(1000)에서 주용단어 500개 선정

3. max length : padding 적용
 - 한 문장을 구성하는 단어 수 지정 속성
 ex) max_length = 100 : 전체 문장을 구성하는 단어수 100개 사용
     1문장 : 150개 단어 구성 -> 50 짤림
     2문장 : 70개 단어 구성 -> 30개 0으로 채움(padding 적용)
"""

# import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # token 생성
from tensorflow.keras.preprocessing.sequence import pad_sequences # padding


# 1. num_words 토큰 생성기
tokenizer = Tokenizer(num_words = 6) # 5개 선정(단어길이 + 1)

texts = ['The dog sat on the table.', 'The dog is my Poodle.']


# 1. 토큰 생성
tokenizer.fit_on_texts(texts) # 텍스트 적용
token = tokenizer.word_index # 토큰 반환
print(token) # dict : {word : index}
# {'the': 1, 'dog': , 2, 'sat': 3, 'on': 4, 'table': 5, 'is': 6, 'my': 7, 'poodle': 8}
print("전체 단어 수 :", len(token)) # 전체 단어 수 : 8

# 2. texts -> 특징(features) 추출 : 희소행렬(sparse matrix)
binary = tokenizer.texts_to_matrix(texts, mode = "binary")
binary # [docs, terms + 1(padding)]
''' 전체 단어 사용
       pad  1   2   3   4   5   6   7   8
array([[0., 1., 1., 1., 1., 1., 0., 0., 0.],
       [0., 1., 1., 0., 0., 0., 1., 1., 1.]])

# num_words = 6 : 5개 단어 사
array([[0., 1., 1., 1., 1., 1.],
       [0., 1., 1., 0., 0., 0.]])
'''

count = tokenizer.texts_to_matrix(texts, mode = "count")
count
'''         1(the)
array([[0., 2., 1., 1., 1., 1., 0., 0., 0.],
       [0., 1., 1., 0., 0., 0., 1., 1., 1.]])

array([[0., 2., 1., 1., 1., 1.],
       [0., 1., 1., 0., 0., 0.]])
'''

# 출현빈도 -> 비율
freq = tokenizer.texts_to_matrix(texts, mode = "freq") # FQ
freq
'''
array([[0.        , 0.33333333, 0.16666667, 0.16666667, 0.16666667,
        0.16666667, 0.        , 0.        , 0.        ],
       [0.        , 0.2       , 0.2       , 0.        , 0.        ,
        0.        , 0.2       , 0.2       , 0.2       ]])
'''

# 출현빈도 -> 비율(TF * 1/DF) : 가장 선호하는 방법
tfidf = tokenizer.texts_to_matrix(texts, mode = "tfidf") # FQ
tfidf
'''                 1(the)
array([[0.        , 0.86490296, 0.51082562, 0.69314718, 0.69314718,
        0.69314718, 0.        , 0.        , 0.        ],
                    1(the)
       [0.        , 0.51082562, 0.51082562, 0.        , 0.        ,
        0.        , 0.69314718, 0.69314718, 0.69314718]])
'''
tfidf.shape # (2, 9) -> (doc, terms + 1) -> (2, 6)

# 3. max length : padding 적용
seq_index = tokenizer.texts_to_sequences(texts) # 정수 index 반환
seq_index
# 1문장 : 6개 단어, 2문장 : 2개
# [[1, 2, 3, 4, 1, 5], [1, 2]]

lens = [len(seq) for seq in seq_index] # [6, 2]
max_length = max(lens) # 6

# padding : max length
x_data = pad_sequences(seq_index, maxlen = max_length)
x_data
'''
array([[1, 2, 3, 4, 1, 5],
       [0, 0, 0, 0, 1, 2]])
'''

x_data = pad_sequences(seq_index, maxlen = 4)
x_data
'''
array([[3, 4, 1, 5],  -> 2개 단어 짤림
       [0, 0, 1, 2]]) -> 2개 단어 채움
'''








