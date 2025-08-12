# =================================================================
# STAGE 1: Builder - 파이썬 패키지 빌드를 위한 스테이지
# =================================================================
# Base 이미지
FROM python:3.11-slim as builder

# 빌드 도구 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    git \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드 및 requirements.txt 복사/설치
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir numpy cython
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade python-magic

# =================================================================
# STAGE 2: Final - 실제 애플리케이션 실행을 위한 스테이지
# =================================================================
# 런타임 이미지
FROM python:3.11-slim

# 런타임에 필요한 시스템 패키지만 설치
RUN apt-get update && apt-get install -y \
    fonts-nanum* \
    fonts-nanum-coding \
    fonts-nanum-extra \
    fontconfig \
    libmagic1 \
    curl \
    libblas3 \
    liblapack3 \
    libjpeg62-turbo \
    libpng16-16 \
    libfreetype6 \
    libffi8 \
    libssl3 \
    libxml2 \
    libxslt1.1 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 한글 폰트 설정
RUN fc-cache -fv
ENV FONTCONFIG_PATH=/etc/fonts

# Python 환경 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 빌드된 패키지들을 런타임으로 복사
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 애플리케이션 파일 복사
WORKDIR /app
COPY . /app/

# 포트 설정
EXPOSE 8501

# 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 실행
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true" ]