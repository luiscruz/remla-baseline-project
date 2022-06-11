FROM python:3.8.13-slim AS model_build

WORKDIR /root/

COPY pyproject.toml setup.py ./
COPY src src
COPY services-shared-folder/data data
COPY reports reports
COPY params.yaml dvc.yaml ./

RUN python -m pip install --upgrade pip &&\
    python -m pip install -r src/requirements.txt

RUN mkdir models &&\
    dvc init --no-scm &&\
    dvc repro

FROM python:3.8.13-slim

WORKDIR /root/

RUN mkdir models
COPY --from=model_build /root/models models
COPY --from=model_build /root/nltk_data nltk_data

COPY src src

COPY requirements.txt params.yaml pyproject.toml setup.py ./
RUN python -m pip install --upgrade pip &&\
    python -m pip install -r src/requirements.txt

EXPOSE 5000