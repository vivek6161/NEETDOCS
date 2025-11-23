# ===============================
# STAGE 1: BUILD ENVIRONMENT
# ===============================
FROM python:3.11-slim as builder

ENV PYTHONUNBUFFERED=1
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libtesseract-dev \
        build-essential \
        cmake \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        gfortran \
        libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

COPY . .

# ===============================
# STAGE 2: RUNTIME IMAGE
# ===============================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PORT=8080
ENV TZ=Asia/Kolkata
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libglib2.0-0 \
        libstdc++6 \
        libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

EXPOSE 8080

CMD ["python3", "-m", "flask", "run", "--port", "8080"]
