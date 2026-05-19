# InterviewMirror-AI patch steps

1. From the project root, pull latest main.

```bash
git pull origin main
```

2. Copy these files into the project root, preserving paths.

```txt
Dockerfile
grpc_server.py
interview.proto
question_client_test.py
requirements_grpc.txt
questions/__init__.py
questions/gemini_question_client.py
questions/schemas.py
```

3. Regenerate gRPC files locally.

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. interview.proto
```

4. Run server.

```bash
python grpc_server.py
```

5. In another terminal, run question generation test.

```bash
python question_client_test.py
```

6. Docker build.

```bash
docker build -t interviewmirror-ai .
```

7. Docker run.

```bash
docker run --env-file .env -p 50051:50051 interviewmirror-ai
```

8. Commit and push.

```bash
git status
git add .
git commit -m "feat: add initial and follow-up question generation"
git pull --rebase origin main
git push origin main
```
