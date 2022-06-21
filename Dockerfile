# FROM python:3.8.13-slim AS model_build

# WORKDIR /root/

# RUN apt-get update; apt-get install -y git

# COPY pyproject.toml setup.py ./
# COPY src src
# # COPY services-shared-folder/data data
# # COPY reports reports
# # COPY params.yaml dvc.yaml ./

# RUN python -m pip install --upgrade pip &&\
#     python -m pip install -r src/requirements.txt

# # RUN mkdir -p data/processed &&\
# #     mkdir -p data/interim &&\
# #     mkdir -p data/external &&\
# #     mkdir models

# RUN git clone -b dvc-versioning https://github.com/Adam-TU/remla-project.git dvc-versioning
# COPY src/training_service/load_key.py ./
# RUN mv load_key.py dvc-versioning

# WORKDIR /root/dvc-versioning

# load google drive api key secret into file for use by DVC
# RUN echo $API_KEY_SECRET > remla-352721-99f80e5bc090.json
# RUN export KEY_FILE=remla-352721-99f80e5bc090.json
# RUN echo $printenv > env.txt
# RUN python load_key.py

# RUN dvc init -f

# # create config entries for gdrive authentication to go automatically 
# # by utilizing the API_KEY_SECRET json credentials
# # add dvc cache remote and link it with json creds (and set as default remote)
# RUN dvc remote add -d dvc-cache-remote gdrive://1pwqW-DruetPFaUBeO2KnnnPwccOZGdZw &&\
#     dvc remote modify dvc-cache-remote gdrive_use_service_account true &&\
#     dvc remote modify dvc-cache-remote --local gdrive_service_account_json_file_path remla-352721-99f80e5bc090.json

# RUN dvc pull

FROM python:3.8.13-slim

WORKDIR /root/

# RUN mkdir models
# COPY --from=model_build /root/dvc-versioning/models models
# COPY --from=model_build /root/dvc-versioning/nltk_data nltk_data

COPY src src

COPY requirements.txt params.yaml pyproject.toml setup.py ./

RUN apt-get update; apt-get install -y git; apt-get install -y curl
RUN python -m pip install --upgrade pip &&\
    python -m pip install -r src/requirements.txt

EXPOSE 5000