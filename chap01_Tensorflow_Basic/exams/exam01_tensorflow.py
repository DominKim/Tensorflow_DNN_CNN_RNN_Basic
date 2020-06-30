'''
문) 두 상수를 정의하고, 사칙연산(+,-,*,/)을 정의하여 결과를 출력하시오.
  조건1> 두 상수 이름 : a, b
  조건2> 변수 이름 : adder,subtract,multiply,divide
  조건3> 출력 : 출력결과 예시 참고
  
<<출력결과>>
a= 100
b= 20
===============
덧셈 = 120
뺄셈 = 80
곱셈 = 2000
나눗셈 = 5.0
'''

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함 

'''프로그램 정의 영역'''

# 상수 정의 
a = tf.constant(100)
b = tf.constant(20) 

# 변수 정의 
adder = tf.Variable(a + b)
subtract = tf.Variable(a - b)
multiply = tf.Variable(a * b)
divide = tf.Variable(a / b)

'''프로그램 실행 영역'''
sess = tf.Session() # # session object 생성 

init = tf.global_variables_initializer()
sess.run(init)
lst = ["덧셈", "뺄셈", "곱셉", "나눗셈"]
lst2 = [adder, subtract, multiply, divide]
lst3 = ["a", "b"]
lst4 = [a, b]

print("<<출력결과>>")
for i, j in zip(lst3, lst4):
    print(i, "=", sess.run(j))
for _ in range(15):
    print("=", end = "")
print()
for i, k in zip(lst, lst2):
    print(i, "=", sess.run(k))
    
1629030*12
19,548,360