# Dockerfile needs to be built from root
FROM python:3.8.13-slim
WORKDIR /app/src

COPY src/training_service/requirements.txt .
RUN apt-get update; apt-get install -y curl
RUN python -m pip install --upgrade pip && python -m pip install -r ./requirements.txt

COPY src/models ./models
COPY src/features ./features
COPY src/preprocess ./preprocess
COPY src/training_service ./training_service
COPY src/gunicorn_config.py src/startup_training.sh ./

WORKDIR /app

COPY src/training_service/remla-352721-99f80e5bc090.json ./
COPY params.yaml dvc.yaml ./
COPY dvc.yaml ./

RUN mkdir models && mkdir reports



