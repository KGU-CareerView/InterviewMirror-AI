FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_grpc.txt .

RUN pip install --no-cache-dir -r requirements_grpc.txt

COPY . .

RUN mkdir -p generated \
    && python -m grpc_tools.protoc \
    -I proto \
    --python_out=generated \
    --grpc_python_out=generated \
    proto/interview.proto \
    && python -c "from pathlib import Path; p=Path('generated/interview_pb2_grpc.py'); s=p.read_text(); s=s.replace('import interview_pb2 as interview__pb2', 'from generated import interview_pb2 as interview__pb2'); p.write_text(s); Path('generated/__init__.py').touch()"

RUN pip install --no-cache-dir --no-deps --no-build-isolation -e .

EXPOSE 50051

CMD ["python", "-m", "app.grpc_server"]
