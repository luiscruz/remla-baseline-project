FROM python:3.8.13-slim AS model_build

WORKDIR /root/

RUN apt-get update; apt-get install -y git

COPY pyproject.toml setup.py ./
COPY src src
COPY services-shared-folder/data data
COPY reports reports
COPY params.yaml dvc.yaml ./

RUN python -m pip install --upgrade pip &&\
    python -m pip install -r src/requirements.txt

RUN mkdir -p data/processed &&\
    mkdir -p data/interim &&\
    mkdir -p data/external &&\
    mkdir models

# load google drive api key secret into file for use by DVC
RUN echo $API_KEY_SECRET > remla-352721-99f80e5bc090.json

RUN git clone -b dvc-versioning https://github.com/Adam-TU/remla-project.git dvc-versioning
RUN mv dvc-versioning/dvc.lock .

RUN dvc init --no-scm -f

# create config entries for gdrive authentication to go automatically 
# by utilizing the API_KEY_SECRET json credentials
# add dvc cache remote and link it with json creds (and set as default remote)
RUN dvc remote add -d dvc-cache-remote gdrive://1pwqW-DruetPFaUBeO2KnnnPwccOZGdZw &&\
    dvc remote modify dvc-cache-remote gdrive_use_service_account true &&\
    dvc remote modify dvc-cache-remote --local gdrive_service_account_json_file_path remla-352721-99f80e5bc090.json

RUN dvc pull

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