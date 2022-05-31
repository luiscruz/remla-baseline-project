# https://www.pybootcamp.com/blog/how-to-write-dockerfile-python-apps/
FROM python:
WORKDIR /root/

RUN pip install -U \
    pip \
    setuptools \
    wheel

COPY requirements.txt ./
RUN pip --no-cache install -r requirements.txt

ARG GIT_HASH
ENV GIT_HASH=${GIT_HASH:-dev}

COPY . .

RUN dvc pull && dvc repro

EXPOSE 8080

ENTRYPOINT python
