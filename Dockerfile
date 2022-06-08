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

# Must be included since we need to store models in it. 
# DVC can raise error if this folder is not present.
RUN mkdir models

# Must be included for DVC to generate and store scores.json file
# Otherwise it would complain about it.
RUN mkdir reports

COPY src src
COPY params.yaml .
COPY dvc.yaml .
COPY data data

# Including .dvc is optional. 
# But if you don't include this, then add `dvc init --no-scp` in the beginning of startup scripts
COPY .dvc .dvc

# .git must be included for dvc repro to execute. 
# Don't think it matters what .git folder to include, as long as it is a valid git folder
COPY .git .git 

RUN python -m pip install --upgrade pip &&\
    pip install -r src/requirements.txt

EXPOSE 5000