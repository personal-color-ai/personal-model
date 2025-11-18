FROM python:3.10-slim

WORKDIR /app

# 1. 시스템 패키지 설치 (OpenCV, GL 라이브러리)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Playwright 브라우저 설치
RUN playwright install --with-deps chromium

# 4. 소스 코드 복사
COPY . .

# 5. 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]