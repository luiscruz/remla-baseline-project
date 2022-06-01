# https://www.pybootcamp.com/blog/how-to-write-dockerfile-python-apps/
FROM python:3.7.10-slim

WORKDIR /root/

# Keep setuptools and wheel up to date
RUN pip install -U \
    setuptools \
    wheel

COPY requirements.txt .
# Keep pip up to date and install packages from requirements.txt
RUN python -m pip install --upgrade pip &&\
    pip install -r requirements.txt

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
