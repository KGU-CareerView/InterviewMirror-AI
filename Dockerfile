FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_grpc.txt .

RUN pip install --no-cache-dir -r requirements_grpc.txt

COPY . .

EXPOSE 50051

CMD ["python", "grpc_server.py"]