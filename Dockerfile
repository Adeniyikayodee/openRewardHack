FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/__init__.py scripts/__init__.py
COPY scripts/fetch_weather.py scripts/fetch_weather.py
COPY scripts/generate_tasks.py scripts/generate_tasks.py

EXPOSE 8000
CMD ["python", "-m", "src.server"]
