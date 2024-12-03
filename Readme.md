## Dockerfile
docker build -t tensorflow .

## 컨테이너 실행
docker run -it --rm --name tensorflow -v $(pwd):/app -w /app/scripts tensorflow python <파일명>




