# https://www.pybootcamp.com/blog/how-to-write-dockerfile-python-apps/
FROM python:3.7.10-slim

WORKDIR /root/

RUN : \
    && apt-get update \
    && DEBIAN_FRONTED=noninteractive apt-get install -y build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv ./venv
ENV PATH=./venv/bin:$PATH

# Keep pip, setuptools and wheel up to date
RUN pip install --upgrade \
    pip \
    setuptools \
    wheel

COPY requirements.txt .
RUN pip install -r requirements.txt

ARG GIT_HASH=dev
ENV GIT_HASH=$GIT_HASH
LABEL git_hash=$GIT_HASH

COPY . .
# RUN dvc pull # TODO: Enable when dvc pull works with authorization

ARG PORT=5000
ENV PORT=$PORT

EXPOSE $PORT

ENTRYPOINT ["python"]
CMD ["src/serve_model.py"]
