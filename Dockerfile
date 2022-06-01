# https://www.pybootcamp.com/blog/how-to-write-dockerfile-python-apps/
FROM python:3.7.10-slim
WORKDIR /root/

RUN pip install -U \
    pip \
    setuptools \
    wheel

COPY requirements.txt ./
RUN pip --no-cache install -r requirements.txt

ARG GIT_HASH=dev
ENV GIT_HASH=$GIT_HASH
LABEL git_hash=$GIT_HASH

COPY . .
RUN dvc pull && dvc repro

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["src/server_model.py"]
