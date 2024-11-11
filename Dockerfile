# 1. Python 3.12.4 이미지로 시작
FROM python:3.12.4

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 파일을 제외하고 복사
COPY . .
RUN rm -rf dataset result yolov8_env

# 4. 패키지 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. 실행 명령
CMD ["python", "code/server/web.py","--host=0.0.0.0"]