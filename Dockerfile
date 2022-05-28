# syntax=docker/dockerfile:1
FROM python:3.8.13-slim AS model_build

WORKDIR /root/

COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .

RUN python -m pip install --upgrade pip &&\
    pip install -r requirements.txt

COPY src src
COPY data data
COPY reports reports

COPY params.yaml .
COPY dvc.yaml .
COPY gateway_nginx.conf .
COPY Makefile .
COPY pre-commit.sh .
COPY test_environment.py .
COPY tox.ini .

RUN mkdir models &&\
    dvc init --no-scm &&\
    dvc repro

FROM python:3.8.13-slim

WORKDIR /root/

RUN mkdir models
COPY --from=model_build /root/models /models
COPY --from=model_build /root/models /models

COPY src src

COPY requirements.txt .
COPY params.yaml .
COPY setup.py .
COPY pyproject.toml .

RUN python -m pip install --upgrade pip &&\
    pip install -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["src/serve_model.py"]
