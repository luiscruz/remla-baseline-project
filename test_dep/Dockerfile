FROM python:3.8.13-slim

WORKDIR /root/

RUN mkdir models
RUN mkdir reports

COPY params.yaml .
COPY dvc.yaml .
COPY src src
COPY data data
COPY .dvc .dvc
COPY .git .git

RUN python -m pip install --upgrade pip &&\
    pip install -r src/requirements.txt

EXPOSE 5000