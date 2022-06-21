#!/usr/bin/env sh

SCRIPT_PATH=$(pwd)
echo $SCRIPT_PATH

# clone dvc-versioning branch and set origin to url with auth token
git clone -b dvc-versioning https://github.com/Adam-TU/remla-project.git dvc-versioning

cp -r src dvc-versioning

cd dvc-versioning
git remote set-url origin https://$GITHUB_ACCESS_TOKEN@github.com/Adam-TU/remla-project.git

git config --global user.email "training@training.com"
git config --global user.name "training-service"

echo $(pwd)

cd $SCRIPT_PATH/src/training_service/

# load google drive api key secret into file for use by DVC
python load_key.py
mv $KEY_FILE $DVC_VERSIONING_PATH

cd $DVC_VERSIONING_PATH

cp -r $SCRIPT_PATH/models $DVC_VERSIONING_PATH
cp -r $SCRIPT_PATH/reports $DVC_VERSIONING_PATH

echo $KEY_FILE >> .gitignore

dvc init -f

# create config entries for gdrive authentication to go automatically 
# by utilizing the API_KEY_SECRET json credentials

# add dvc cache remote and link it with json creds (and set as default remote)
dvc remote add -d dvc-cache-remote gdrive://1pwqW-DruetPFaUBeO2KnnnPwccOZGdZw
dvc remote modify dvc-cache-remote gdrive_use_service_account true
dvc remote modify dvc-cache-remote --local gdrive_service_account_json_file_path $KEY_FILE
dvc remote modify dvc-cache-remote --local gdrive_service_account_user_email 'remla-project@remla-352721.iam.gserviceaccount.com'

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

cd $SCRIPT_PATH

gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 --timeout 600 src.$APP_MODULE:app
