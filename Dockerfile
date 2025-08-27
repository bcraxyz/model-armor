# Stage 1
FROM python:3.12-slim AS builder

WORKDIR /install
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Stage 2
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /usr/src/app

COPY --from=builder /install /usr/local
COPY . .

RUN useradd -m appuser
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "./cloudrun_app.py", "--server.port=8501", "--server.enableXsrfProtection=false"]
