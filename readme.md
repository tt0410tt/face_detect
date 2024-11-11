# 딥러닝 기반 안면 비식별화 솔루션

# 목차
1. 프로젝트 개요
2. 기능
3. 서버 조건 및 설치 방법
4. 사용법
5. 기술 스택
6. 라이선스
7. 문의

# 1. 프로젝트 개요
- Python과 딥러닝 기술을 활용한 영상 안면 비식별화 웹 애플리케이션입니다.  
사용자가 업로드한 영상을 분석하여 얼굴을 인식하고 비식별화를 진행한 후 결과를 웹 인터페이스를 통해 제공합니다.

# 2. 기능
## 핵심기능 
1. 안면 인식 및 추적
    - 동영상 프레임에서 사람의 얼굴을 자동으로 탐지 및 추적    
    - 다수의 얼굴 동시 인식

2. 안면 비식별화(사각형, 타원형)
    - 블러링 : 얼굴에 블러 효과 적용
    - 모자이크 : 얼굴에 모자이크 효과 적용

## 기능
1. 사용자 인터페이스(UI)
    - 웹 기반 UI 제공
    - 처리 결과 다운로드

# 3. 서버 조건 및 설치 방법
## 서버 조건
- anaconda or 도커 사용가능자(도커 환경 권장)  
- requests.txt의 모든 라이브러리를 사용가능한 환경  
- git 사용가능자  

## 설치 방법
- git, docker 사용 기준(권장)
    1. git 클론할 폴더를 만든다.  
    2. 터미널에서 그 위치를 찾아간다.  
    3. 터미널에서  
    git clone https://github.com/tt0410tt/face_detect.git  
    를 입력한다.  
    이후 아래 커멘드를 순서대로 터미널에 입력한다.
    4. cd face_detect  
    5. docker build -t [프로젝트이름] .  
    6. docker run -p 5000:5000 [프로젝트이름]  


# 4. 사용법
- git, docker 사용 기준(권장)  
    1. 크롬(권장)을 켜서 주소창에 http://localhost:5000/ 입력한다.  
    2. 메인화면에서 파일선택을 누른다.  
    ![alt text](readme_images/n1.png)  
    3. 모달창이 뜨면 안면 비식별화가 필요한 mp4 파일을 선택한다.  
    4. 제출을 누른다.  
    5. 화면에서 4가지 비식별화 솔루션 중 한가지를 선택한다.  
    ![alt text](readme_images/n2.png)  
    6. 비식별화를 진행하는동안 대기한다.  
    7. 다운로드를받는다.  
    ![alt text](readme_images/n3.png)  
    

# 5. 기술 스택
- 프론트엔드: HTML, CSS, JavaScript
- 백엔드: Python, Flask
- 딥러닝 라이브러리

# 6. 라이선스
라이선스
이 프로젝트는 MIT 라이선스를 따릅니다. 
The MIT License (MIT)

Copyright (c) <year> <copyright holders>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# 7. 문의
이메일: tt0410tt@naver.com  
GitHub: tt0410tt