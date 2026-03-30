FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/
COPY models/ ./models/

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PORT=8002

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8002/sentiment/health || exit 1

CMD ["python", "-m", "sentiment_api.main"]
