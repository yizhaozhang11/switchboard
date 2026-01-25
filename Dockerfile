FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BOT_DATA_DIR=/data

WORKDIR /app

RUN mkdir -p /data

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app

CMD ["python", "-m", "app.main"]
