FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# runtime needs scripts/fetch_weather (used by src/feasibility); the rest of
# scripts/ is generation-time only and intentionally left out.
COPY scripts/__init__.py scripts/__init__.py
COPY scripts/fetch_weather.py scripts/fetch_weather.py

# bake the published dataset into the image so the env runs standalone with
# no host volume mount. ORWD_DATA_DIR can still be overridden at runtime to
# point at a Files-tab mount on the OpenReward platform.
COPY data/tasks.parquet /app/data/tasks.parquet
ENV ORWD_DATA_DIR=/app/data

EXPOSE 8080
ENV PORT=8080
CMD ["python", "-m", "src"]
