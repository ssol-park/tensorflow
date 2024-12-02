## Dockerfile
docker build -t tensorflow .

## 컨테이너 실행
docker run -it --name tensorflow -v $(pwd):/app -w /app tensorflow