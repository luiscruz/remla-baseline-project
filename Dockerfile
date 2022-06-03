FROM python:3.9.12-slim

WORKDIR /root/

RUN apt-get update &&\
    apt-get install --no-install-recommends -y gcc g++

COPY requirements.txt .
COPY setup.py .
COPY src src

RUN python -m pip install --upgrade pip &&\
    pip install -r requirements.txt &&\
    pip install -e .[all]

COPY .pylintrc .
COPY tests tests

# Copy necessary files and folders for dvc
COPY .dvc .dvc
COPY dvc.yaml .
COPY dvc.lock .
COPY .git .git
COPY data data
COPY models models

# Dynamic interaction with the model
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["src/models/serve_model.py"]
