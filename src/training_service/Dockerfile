# Dockerfile needs to be built from root
FROM python:3.8.13-slim
WORKDIR /app/src
COPY src/training_service/* .
COPY src/gunicorn_config.py .
COPY src/startup.sh .
COPY params.yaml .

WORKDIR /app

RUN apt-get update; apt-get install -y curl
RUN python -m pip install --upgrade pip && python -m pip install -r src/requirements.txt

