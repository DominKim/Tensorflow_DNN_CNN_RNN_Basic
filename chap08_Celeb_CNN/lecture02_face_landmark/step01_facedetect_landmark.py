# -*- coding: utf-8 -*-
"""
image 얼굴인식과 68 point landmark 인식 

1. Anaconda python 3.7 설치 : git hub API 이용
1) dlib 설치
> conda install -c conda-forge dlib 
2) 68 point landmark 학습 data 다운로드
http://dlib.net/files
shape_predictor_68_face_landmarks.dat
"""

import dlib # face detection/face landmark
from skimage import io # image read/save
import os # dir 생성/이동/경로 
from glob import glob # dir 패턴검색(*jpg)

# 현재 경로 기준  
fpath = os.getcwd() + "/" + "images" # raw image 위치 
fpath2 = os.getcwd() + "/"+ "croped_images" # croped image 저장 위치

# 68 landmark 학습 data 
predictor = "shape_predictor_68_face_landmarks.dat"

# hog 얼굴 인식기(알고리즘)
face_detector = dlib.get_frontal_face_detector()
# 68 landmark 인식기 
face_68_landmark = dlib.shape_predictor(predictor)


i = 0
for file in glob(fpath+"/*.jpg") :   
    image = io.imread(file)
    print(image.shape) 
    
    # image 출력장 표시 
    win = dlib.image_window()
    win.set_image(image)
        
    detected = face_detector(image)
    print('인식한 face size =', len(detected))
    
    i += 1
    for face in detected : 
        # 감지된 image 사각점 좌표 
        print(face) # [(141, 171) : 왼쪽 위  (409, 439) : 오른쪽 아래]
        
        print(f'왼쪽 : {face.left()}, 위 : {face.top()}, 오른쪽 : {face.right()}, 아래 : {face.bottom()}')
        # 왼쪽 : 141, 위 : 171, 오른쪽 : 409, 아래 : 439
        
        
        # 이미지 출력장에 face 사각점 좌표 겹치기 
        win.add_overlay(face)
        
        # 이미지 face 사각점안에 68 point 겹치기
        face_landmark = face_68_landmark(image, face)
        win.add_overlay(face_landmark)
        
        # 크롭(crop) : 얼굴 부분만 자르기 : image[h, w]
        crop = image[face.top():face.bottom(), face.left():face.right()]
        
        # croped image save
        io.imsave(fpath2 + "/croped"+str(i)+".jpg", crop)
