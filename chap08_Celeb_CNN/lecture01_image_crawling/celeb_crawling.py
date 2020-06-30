# -*- coding: utf-8 -*-
"""
pip install selenium
chromedriver download 버전 : 83.0.4103.97(64비트)
"""

from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.request import urlretrieve # server image -> local save
import os # dir 경로/생성/이동
import numpy as np

pwd = os.getcwd() # 현재 경로 
print(pwd)
    

def celeb_crawler(name) : 
    # 1. dirver 경로/파일 지정 
    driver = webdriver.Chrome("C:\\ITWILL\\6_Tensorflow\\workspace\\chap08_Celeb_CNN\\lecture01_image_crawling\\chromedriver.exe")
    
    # 2. 이미지 검색 url 
    driver.get("https://www.google.co.kr/imghp?hl=ko")
    
    # 3. 검색 입력상자 tag -> 검색어 입력   
    search_box = driver.find_element_by_name('q') # element 이름 찾기 
    '''
    find_element_by_name('name') : element name 속성 이름으로 찾기 
    find_element_by_id('id') : element id 속성의 이름으로 찾기 
    find_element_by_xpath('//tag[@속성='값']/tag') : tab의 계층적 구조로 찾기 
    '''
    search_box.send_keys(name) # 검색어 키보드 입력(driver 객체 member) 
    driver.implicitly_wait(3) # 3초 대기(자원 loading)
    
    # 4. [검색] 버튼 클릭 ("//tag[@attr='value']/sub element")
    driver.find_element_by_xpath("//div[@id='sbtc']/button").click() 
    
    '''
    <div jsname="BN6WQc">
       <div data-ri="0"> : 첫번째 image
       <div data-ri="4"> : 다섯번째 image
    '''    
    image_url = []
     
    i=0 # 첫번째 이미지    
    base = f"//div[@data-ri='{i}']"
    driver.find_element_by_xpath(base).click() # image click
    # click image url 
    src = driver.page_source # 현재 page html source
    html = BeautifulSoup(src, "html.parser")
    img_tag = html.select("img[class='rg_i Q4LuWd']") # list
    print(img_tag)
    
    for tag in img_tag : 
        if 'data-src' in str(tag) :
            url = tag['data-src'] # dict
            image_url.append(url)
    
    # image url 생성             
    print(len(image_url)) # 6
    image_url = np.unique(image_url) # 중복 url  삭제 
    print(image_url)
    
    # url -> image save
    for i in range(len(image_url)) :
        try : # 예외처리 : server file 없음 예외처리 
            file_name = "test"+str(i+1)+".jpg" # test1.jsp
            # server image -> file save
            urlretrieve(image_url[i], filename=file_name)
        except :
            print('해당 url에 image 없음 : ', i+1)        
            
    driver.close()
    
# 함수 test     
#celeb_crawler("강호동")   

# 함수 호출 : 각 셀럼당 6개 이미지 
namelist = ["하정우","송강호","전지현"]

for name in namelist :
    pwd = os.getcwd() # 현재 경로 
    os.mkdir(name) # 현재 위치에 폴더 생성 
    os.chdir(pwd+"/"+name) # lecture/하정우
    celeb_crawler(name) # image crawling
    os.chdir(pwd) # 원래 위치 이동 

    







 
    
    