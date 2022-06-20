#!/usr/bin/env sh

# load google drive api key secret into file for use by DVC
python src/training_service/load_key.py

# clone dvc-versioning branch
git clone -b dvc-versioning https://github.com/Adam-TU/remla-project.git

dvc init -f

# create config entries for gdrive authentication to go automatically 
# by utilizing the API_KEY_SECRET json credentials

# add dvc cache remote and link it with json creds (and set as default remote)
dvc remote add -d dvc-cache-remote gdrive://1pwqW-DruetPFaUBeO2KnnnPwccOZGdZw
dvc remote modify dvc-cache-remote gdrive_use_service_account true
dvc remote modify dvc-cache-remote --local gdrive_service_account_json_file_path $KEY_FILE
dvc remote modify dvc-cache-remote --local gdrive_service_account_user_email 'remla-project@remla-352721.iam.gserviceaccount.com'


# add remote for fetching data from google drive and link it with json creds
# dvc remote add data-remote gdrive://1fgdFNFF0YE-U0GHYPcpQA3I7erSSQ9iU
# dvc remote modify data-remote gdrive_use_service_account true
# dvc remote modify data-remote --local gdrive_service_account_json_file_path remla-352721-99f80e5bc090.json

echo $DVC_CACHE_PATH

dvc cache dir $DVC_CACHE_PATH
dvc config cache.shared group
dvc config cache.type symlink

# For DVC ADD command so that it fetches the raw data from shared storage
# NOTE: this folder must be removed and re-created when the data needs to be fetched again.
# mkdir -p data/raw

# recreating other required folders for dvc repro to succeed
mkdir -p data/processed
mkdir -p data/interim
mkdir -p data/external

# fetching the input data for dvc repro from shared folder
# dvc add $SHARED_DATA_PATH/raw/test.tsv -o data/raw/test.tsv
# dvc add $SHARED_DATA_PATH/raw/train.tsv -o data/raw/train.tsv
# dvc add $SHARED_DATA_PATH/raw/validation.tsv -o data/raw/validation.tsv
dvc add $SHARED_DATA_PATH/raw/ -o data/raw/

# reproduce pipeline to create the models and output data
# this also updates the cache if something has changed in the data
dvc repro

gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 --timeout 600 src.$APP_MODULE:app
