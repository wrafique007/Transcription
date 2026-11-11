FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY transcriber ./transcriber
ENV STT_MODEL=small STT_DEVICE=auto STT_WORKERS=1
EXPOSE 8000
CMD ["uvicorn", "transcriber.api:app", "--host", "0.0.0.0", "--port", "8000"]
