# https://www.pybootcamp.com/blog/how-to-write-dockerfile-python-apps/
FROM python:3.7.10-slim

WORKDIR /root/

# Install build-essential package to let the tensorflow-data-validation package work
RUN : \
    && apt-get update \
    && DEBIAN_FRONTED=noninteractive apt-get install -y build-essential curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Keep pip, setuptools and wheel up to date
RUN pip install --upgrade \
    pip \
    setuptools \
    wheel

RUN curl -sSL -o install-poetry.py https://install.python-poetry.org
RUN python3 install-poetry.py --pre
ENV PATH=/root/.local/bin/:$PATH
RUN poetry config virtualenvs.in-project true
RUN poetry install
RUN echo "source /root/.venv/bin/activate" >> /root/.profile
SHELL ["sh", "-lc"]

ARG GIT_HASH=dev
ENV GIT_HASH=$GIT_HASH
LABEL git_hash=$GIT_HASH

COPY . .
RUN dvc pull

ARG PORT=5000
ENV PORT=$PORT

EXPOSE $PORT

ENTRYPOINT ["python"]
CMD ["src/serve_model.py"]
