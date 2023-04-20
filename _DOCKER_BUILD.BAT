docker build --no-cache -t ml-classification -f Dockerfile .
docker run -p 8501:8501 --name ml-classification ml-classification