# -*- coding: utf-8 -*-
"""
step06_variable_assign.py

난수 상수 생성 함수 : 정규분포난수, 균등분포 난수
tf.Variable(난수 상수) -> 변수 값 수정
"""

import tensorflow.compat.v1 as tf # ver 2.x -> ver 1.x
tf.disable_v2_behavior() # ver2.x 사용

# 난수
num = tf.constant(10.0)

# 0차원(scala) 변수
var = tf.Variable(num + 20.0) # 상수 + 상수 = scala
print("var =", var) # var = <tf.Variable 'Variable_10:0' shape=() dtype=float32_ref>

# 1차원 변수
var1d = tf.Variable(tf.random_normal([3])) # 1차원 : [n]
print("var1d =", var1d) # var1d = <tf.Variable 'Variable_12:0' shape=(3,) dtype=float32_ref>

# 2차원 변수
var2d = tf.Variable(tf.random_uniform([3, 2])) # 2차원 : [row,col]
print("var2d =", var2d) # var2d = <tf.Variable 'Variable_15:0' shape=(3, 2) dtype=float32_ref>

# 3차원 변수
var3d = tf.Variable(tf.random_normal([3, 2, 4])) # 3차원 : [side,row,ccol]
print("var3d =", var3d) # var3d = <tf.Variable 'Variable_37:0' shape=(3, 2, 4) dtype=float32_ref>

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) # 변수 초기화(초기값 할당) : var, var1d, var2d
    
    print("var =", sess.run(var)) # var = 30.0
    print("var1d =",sess.run(var1d))
    # var1d = [0.18581896 1.9806285  0.30180988]
    print("var2d =",sess.run(var2d))
    '''
    var2d = [[0.28200257 0.32071364]
             [0.16617727 0.20922947]
             [0.90859985 0.7568222 ]]
    '''
    
    # 변수의 값 수정
    var1d_data = [0.1, 0.2, 0.3]
    print("var1d_assign_add =", sess.run(var1d.assign_add(var1d_data)))
    # var1d_assign_add = [0.28581896 2.1806285  0.60180986]
    print("var1d_assign =", sess.run(var1d.assign(var1d_data)))
    # var1d_assign = [0.1 0.2 0.3]
    
    print("var3d =", sess.run(var3d))
    '''
    var3d = [[[-0.4977221   0.88293463 -0.30572122  1.2777227 ]
              [ 1.2326258   0.09313537 -0.52000874  0.27917522]]

             [[-1.3138567   0.6868766  -0.4141482   0.31458348]
              [ 0.69125706  0.7964648  -0.739175    0.4504485 ]]

             [[-0.2753107   1.5733316  -2.0576172   0.9670806 ]
              [-1.1065412  -1.6314406  -0.85542816  0.08698115]]]
    '''
    var3d_re = sess.run(var3d)
    print(var3d_re[0].sum()) # 1면 : 합계 -10.739544
    print(var3d_re[0, 0].mean()) # 1면, 1행 : 평균 -0.8747419
    
    # 24개 균등포난수를 생성하여 var3d 변수에 값을 수정하시오
    var3d_data = tf.random_uniform([3, 2, 4])
    print("var3d_assign =", sess.run(var3d.assign(var3d_data)))
    '''
    var3d_assign = [[[0.5184742  0.135059   0.53484106 0.05304658]
                     [0.9525733  0.85945225 0.85705745 0.22261691]]
                   
                    [[0.6417128  0.94782865 0.10895514 0.05368257]
                     [0.8589392  0.7626581  0.343722   0.17921674]]
                   
                    [[0.12374485 0.54701257 0.14766788 0.19570863]
                     [0.9646692  0.7847116  0.13286352 0.31502736]]]
    '''
    