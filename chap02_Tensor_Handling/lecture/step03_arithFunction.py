'''
step03_arithFunction.py

주요 수학 관련 함수 
tf.add() -> tf.math.add() 변경 
tf.subtract() -> tf.math.subtract() 변경 
tf.multiply() -> tf.math.multiply() 변경 
tf.div() -> tf.math.divide() 변경 
tf.mod() : 나머지 -> tf.math.mod() 변경 
tf.abs() : 절대값 -> tf.math.abs() 변경 
tf.square() : 제곱  -> tf.math.square() 변경
tf.sqrt() : 제곱근  -> tf.math.sqrt() 변경
tf.round() : 반올림  -> tf.math.round() 변경
tf.pow() : 거듭제곱 -> tf.math.pow() 변경
tf.exp() : 지수값 -> tf.math.exp() 변경
tf.log() : 로그값 -> tf.math.log() 변경
'''

import tensorflow as tf

x = tf.constant([1,2,-3,4])
y = tf.constant([5,6,7,8])


# 덧셈/뺄샘/나눗셈/곱셈
print(tf.math.add(x, y, name='adder'))
print(tf.math.subtract(x, y, name='subtract'))
print(tf.math.multiply(x, y, name='multiply'))
print(tf.math.divide(x, y, name='divide'))
print(tf.math.mod(x, y, name='mod')) # [1 2 4 4]
# print(tf.mod(x,y,name="mod")) # ver 1.x

# 음수, 부호 반환 
print('tf.neg=', tf.math.negative(x))
print('tf.sign=', tf.math.sign(x)) 

# 제곱/제곱근
print(tf.math.abs(x)) # [1 2 3 4]
print(tf.math.square(x)) # [ 1  4  9 16]
print(tf.math.sqrt([4.0, 9.0, 6.0])) 

# 지수와 로그 
print(tf.math.exp(1.0)) # e = 2.7182817
print(tf.math.exp(2.0)) # tf.exp()
print(tf.math.log(8.0)) # tf.log()

