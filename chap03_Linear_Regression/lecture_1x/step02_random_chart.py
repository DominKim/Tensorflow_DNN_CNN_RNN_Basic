import tensorflow as tf # ver1.x

import matplotlib.pyplot as plt # chart
import seaborn as sns

#######################
# 1. 난수 생성 함수 
#######################
''' 
 tf.random_normal(shape, mean, stddev) : 평균,표준편차 적용 정규분포
 -> ver2.0 : tf.random.normal(shape, mean, stddev)
 tf.truncated_normal(shape, mean, stddev) : 표준편차의 2배 수보다 큰 값은 제거하여 정규분포 생성 
 -> ver2.0 : tf.truncated.normal(shape, mean, stddev) 
 tf.random_uniform(shape, minval, maxval) : 균등분포 난수 생성
 -> ver2.0 : tf.random.uniform(shape, minval, maxval) 
 tf.random_shuffle(value) : 첫 번째 차원을 기준으로 텐서의 원소 섞기
 -> ver2.0 : tf.random.shuffle(value)
 tf.set_random_seed(seed) : 난수 seed값 설정 
 -> ver2.0 : tf.random.set_seed(seed)
'''

# 2행3열 구조의 표준정규분포를 따르는 난수 생성  
norm = tf.random.normal([2,3], mean=0, stddev=1)
print(norm) # 객체 보기 

#help(tf.random_normal) # mean, stddev 생략 시 default : 0과 1
tf.random.set_seed(123) # 항상 동일한 난수 생성 
norm2 = tf.random.normal([2,3])
print(norm2) # 객체 보기 

matrix = [[1,2], [3,4], [5,6]] # # (3, 2)
cont = tf.constant(matrix)
shuff = tf.random.shuffle(cont)

print(norm)
print(norm2)    
print(shuff) # 첫번째 차원(3행 단위로 섞음) 


####################################
# 2. 정규분포, 균등분포 차트 시각화
####################################

# 정규분포(평균:0, 표준편차:2) 
norm = tf.random.normal([100], mean=0, stddev=2) 
plt.hist(norm, normed=True)
plt.show()
 
# 균등분포(0~1) 
uniform = tf.random.uniform([100], minval=0,maxval=1) 
plt.hist(uniform, normed=True)
sns.distplot(uniform, kde = True,rug = True)
plt.show() 


# seed 유무에 따른 난수 생성  
tf.random.set_seed(123) 
a = tf.random.uniform([1], seed=123) # seed 있음  
b = tf.random.uniform([1]) # seed 없음 

print("a = {}".format(a)) 
print("b = {}".format(b)) 