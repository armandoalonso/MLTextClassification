docker build -t streamlit-ml -f Dockerfile .
docker run -p 8501:8501 streamlit-ml