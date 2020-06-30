"""
Entropy : 일반 용어 
 - 확률분포에 p에 대한 불확실성의 측정 지수 
 - 이 값이 클 수록 일정한 방향성과 규칙성이 없는 무질서(chaos)를 의미

Cross Entropy : 통계 용어 
 - 두 확률변수 x, y에서 x를 관찰한 후 y에 대한 불확실성을 줄인다.
   -> 역전파 알고리즘에서 가중치(w)와 상수(b) 조정 -> 오차(cost) 줄임  
 - 분류기에서 cost 함수로 이용(정답과 예측치의 오차 계산)  
 - Entropy = -sum(Y * log(model))
"""

import numpy as np

# 1. 불확실성이 큰 경우(p1: 앞면, p2: 뒷면)
p1 = 0.5; p2 = 0.5

entropy = -p1 * np.log2(p1) + -p2 * np.log2(p2)
print('entropy =', entropy) # entropy = 1.0

# p1 -> Y, p2 -> model 
Y = p1; model = p2
entropy = -(Y * np.log2(Y) + model * np.log2(model)) # 공통부호 정리
print('entropy =', entropy) # entropy = 1.0



# 2. 불확실성이 작은 경우(x1: 앞면, x2: 뒷면) 
p1 = 0.9; p2 = 0.1

entropy = -p1 * np.log2(p1) + -p2 * np.log2(p2)
print('entropy =', entropy) # entropy = 0.4689955935892812

# p1 -> Y, p2 -> model 
Y = p1; model = p2
entropy = -(Y * np.log2(Y) + model * np.log2(model)) # 공통부호 정리
print('entropy =', entropy) # entropy = 0.4689955935892812

