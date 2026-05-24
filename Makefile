PYTHON   := venv/bin/python
PROTO    := proto/interview.proto
GEN_DIR  := generated

.PHONY: help proto install run dev clean docker

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  proto    Generate gRPC stubs from proto/interview.proto"
	@echo "  install  Install dependencies into venv"
	@echo "  run      Generate proto stubs then start the gRPC server"
	@echo "  dev      Same as run, with hot-reload via watchfiles"
	@echo "  docker   Build and run the application in a Docker container"
	@echo "  clean    Remove generated stubs and __pycache__"

proto:
	@echo "[proto] Generating stubs..."
	@mkdir -p $(GEN_DIR)
	$(PYTHON) -m grpc_tools.protoc \
		-I proto \
		--python_out=$(GEN_DIR) \
		--grpc_python_out=$(GEN_DIR) \
		$(PROTO)
	@touch $(GEN_DIR)/__init__.py
	@# Fix absolute import in generated grpc stub
	@sed -i 's/^import interview_pb2 as interview__pb2/from generated import interview_pb2 as interview__pb2/' \
		$(GEN_DIR)/interview_pb2_grpc.py
	@echo "[proto] Done -> $(GEN_DIR)/"

install:
	@echo "[install] Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements_grpc.txt
	$(PYTHON) -m pip install --no-deps --no-build-isolation -e .
	@echo "[install] Done"

run: proto
	@echo "[run] Starting gRPC server..."
	$(PYTHON) -m app.grpc_server

dev: proto
	@echo "[dev] Starting gRPC server with auto-reload..."
	$(PYTHON) -m watchfiles "python -m app.grpc_server" app proto

docker:
	@echo "[docker] Building Docker image..."
	docker build -t interview-ai .
	@echo "[docker] Running Docker container..."
	docker run --env-file .env -p 50051:50051 --name interview-ai interview-ai

clean:
	@echo "[clean] Removing generated files..."
	rm -f $(GEN_DIR)/interview_pb2.py $(GEN_DIR)/interview_pb2_grpc.py
	find . -type d -name __pycache__ -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "[clean] Done"
