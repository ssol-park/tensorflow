# Python 3.10 slim 이미지를 베이스로 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip

# 의존성 파일 복사 및 설치
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# 애플리케이션 코드 복사
COPY scripts/ /app/scripts/

# 실행 명령어
# CMD ["python", "test.py"]
