FROM python:3.9.12-slim

WORKDIR /root/

RUN apt-get update &&\
    apt-get install -y gcc

COPY requirements.txt .
COPY setup.py .
COPY .pylintrc .
COPY src src

RUN python -m pip install --upgrade pip &&\
    pip install -r requirements.txt &&\
    pip install -e .[linter]

# TODO: Add entrypoint to ML application here
