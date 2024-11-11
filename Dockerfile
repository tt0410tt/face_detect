# 1. 아나콘다 이미지로 시작
FROM continuumio/anaconda3

# 2. 필요한 시스템 패키지 설치 (libGL 포함)
RUN apt-get update && \
    apt-get install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 새로운 Conda 환경 생성
RUN conda create -n face_env python=3.9 -y

# 5. Conda 환경을 사용하여 필요한 패키지 설치
# `torch`는 `pip`로 설치하고 나머지 패키지는 `conda`로 설치
RUN /bin/bash -c "source activate face_env && \
    conda install -c conda-forge flask matplotlib numpy pillow ultralytics werkzeug -y && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

# 6. 환경 변수를 통해 conda 환경 활성화
ENV PATH="/opt/conda/envs/face_env/bin:$PATH"
ENV CONDA_DEFAULT_ENV="face_env"

# 7. 필요한 파일을 복사하고 불필요한 폴더 삭제
COPY . .
RUN rm -rf dataset result yolov8_env

# 8. 서버 실행 명령
<<<<<<< HEAD
CMD ["python", "code/server/web.py", "--host=0.0.0.0"]
=======
CMD ["python", "code/server/web.py", "--host=0.0.0.0"]
>>>>>>> 386ca96beebdbc8493e0587e78d036a9a9467e34
