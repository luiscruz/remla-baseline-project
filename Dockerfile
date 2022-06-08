# syntax=docker/dockerfile:1
# FROM python:3.8.13-slim AS model_build

# WORKDIR /root/

# COPY requirements.txt .
# COPY setup.py .
# COPY pyproject.toml .

# RUN python -m pip install --upgrade pip &&\
#     pip install -r requirements.txt

# COPY src src
# COPY data data
# COPY reports reports

# RUN echo $(pwd)
# RUN echo $(ls /root/src/data)

# COPY params.yaml .
# COPY dvc.yaml .

# RUN mkdir models &&\
#     dvc init --no-scm &&\
#     dvc repro

FROM python:3.8.13-slim

WORKDIR /root/

RUN mkdir models
RUN mkdir reports

COPY src src
COPY params.yaml .
COPY dvc.yaml .
COPY data data
COPY .dvc .dvc
COPY .git .git

RUN python -m pip install --upgrade pip &&\
    pip install -r src/requirements.txt

EXPOSE 5000