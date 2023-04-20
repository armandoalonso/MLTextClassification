FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements.txt

RUN python -m spacy download en_core_web_lg

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "START.py", "--server.port=8501", "--server.maxUploadSize=1028", "--server.maxMessageSize=1028"]