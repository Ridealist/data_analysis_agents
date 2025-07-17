# Base 이미지
FROM python:3.11-slim

# 2. 작업 디렉토리를 /app으로 설정
WORKDIR /app

# 3. 의존성 패키지 설치
RUN apt-get update && apt-get install -y \
    libmagic1 \
    build-essential cmake python3-dev \
    curl \
    software-properties-common \
    git \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# 4-1. 현재 디렉토리의 requirements.txt 파일을 컨테이너의 /app으로 복사
COPY requirements.txt /app/

RUN pip install numpy cython   

# 4-2. requirements.txt에 명시된 패키지 설치
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -r requirements.txt

# 4-3. reinstall python-magic
RUN pip install --upgrade python-magic

# 5. 실행할 streamlit_app.py를 컨테이너 /app으로 복사
COPY . /app/

# 6. 컨테이너가 수신할 포트 오픈
EXPOSE 8501

# 7. 컨테이너 헬스 체크
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

#8. 실행될 컨테이너 구성
ENTRYPOINT [ "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0" ]