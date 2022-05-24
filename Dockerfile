FROM python:3.8.4-slim

COPY requirements.txt .
COPY src src
COPY data data

RUN mkdir output && python -m pip install --upgrade pip && pip install -r requirements.txt
RUN python -m nltk.downloader -d /usr/share/nltk_data all
RUN python src/text_preprocessing.py && python src/vectorization.py && python src/model_training.py && python src/evaluation.py

EXPOSE 8080