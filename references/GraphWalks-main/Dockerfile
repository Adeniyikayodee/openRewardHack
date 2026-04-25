FROM python:3.11-slim

# Update and install basic dependencies
RUN apt update && apt upgrade -y && apt install -y curl

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (NOT data.parquet - that goes to /orwd_data/)
COPY graphwalks.py .
COPY server.py .

# Expose port
EXPOSE 8080

# Start the server
CMD ["python", "server.py"]
