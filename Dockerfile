# ---------- Stage 1: Builder ----------
FROM python:3.12 AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim

WORKDIR /app

# Копіюємо вже встановлені пакети
COPY --from=builder /usr/local /usr/local

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/train.py"]