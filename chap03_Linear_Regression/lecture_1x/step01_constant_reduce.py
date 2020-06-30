'''
numpy vs tensorflow
 1. 상수 생성 함수 
 2. 차원축소  수학 연산
'''

import tensorflow as tf # ver1.x


'''
1. 상수 생성 함수
 tf.constant(value, dtype, shape) : 지정한 값(value)으로 상수 텐서 생성   
 tf.zeros(shape, dtype) : 모양과 타입으로 모든 원소가 0으로 초기화된 텐서 생성 
 tf.ones(shape, dtype) : 모양과 타입으로 모든 원소가 1로 초기화된 텐서 생성
 tf.identity(input) : 내용과 모양이 동일한 텐서 생성   
 tf.fill(dims, value) : 주어진 scalar값으로 초기화된 텐서 생성 
 tf.tuple(tensor) : 여러 개의 tensor list로 묶기 
 tf.linspace(start, stop, num) : start~stop 범위에서 num수 만큼 텐서 생성  
 tf.range(start, limit, delta) : 시작점, 종료점, 차이 이용 텐서 생성 
'''

a = tf.zeros( (2, 3) )
print(a) # sess.run()


b = tf.ones( (2, 3) )
print(b)

c = tf.fill((2, 3), 5) # (shape, value)
print(c) # sess.run(c)

d = tf.constant(10, tf.int32, (2, 3))
print(d)

e = tf.linspace(15.2, 22.3, 5) # (start, step, size)
print(e)  # [15.2   16.975 18.75  20.525 22.3  ]
16.975-15.2 # 1.775
18.75-16.975 # 1.775

f = tf.range(10, 1.5, -2.5) # (start, stop, delta)
print(f)

print('\n차원축소 관련 함수')
'''
2. 차원축소 수학 연산
 tf.reduce_sum(input_tensor, reduction_indices) : 지정한 차원을 대상으로 원소들 덧셈
 tf.reduce_mean(input_tensor, reduction_indices) : 지정한 차원을 대상으로 원소들 평균
 tf.reduce_prod(input_tensor, reduction_indices) : 지정한 차원을 대상으로 원소들 곱셈
 tf.reduce_min(input_tensor, reduction_indices) : 지정한 차원을 대상으로 최솟값 계산
 tf.reduce_max(input_tensor, reduction_indices) : 지정한 차원을 대상으로 최댓값 계산
 tf.reduce_all(input_tensor) : tensor 원소가 전부 True -> True 반환
 tf.reduce_any(input_tensro) : tensor 원소가 하나라도 True -> True 반환  
'''
'''
[[5 5 5]
 [5 5 5]]
'''


print(tf.reduce_sum(c, axis=0)) # 행축 합계:열 단위  
print(tf.reduce_sum(c, axis=1)) # 열축 합계:행 단위   

data = [[1., 1.], [2., 2.]]
x = tf.constant(data) # 축생략은 전체 모든원소를 대상으로 한다.
print(tf.reduce_mean(x)) # 1.5
print(tf.reduce_max(x)) # 2.0 

bool_data = [[True, True], [False, False]] 
print(tf.reduce_all(bool_data)) 
print(tf.reduce_any(bool_data)) 


