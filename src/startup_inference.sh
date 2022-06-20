# #!/usr/bin/env sh
rm -rf $PROMETHEUS_MULTIPROC_DIR
mkdir $PROMETHEUS_MULTIPROC_DIR

# load google drive api key secret into file for use by DVC
python src/training_service/load_key.py

# clone dvc-versioning branch and set origin to url with auth token
git clone -b dvc-versioning https://github.com/Adam-TU/remla-project.git dvc-versioning
cd dvc-versioning
git remote set-url origin https://$GITHUB_ACCESS_TOKEN@github.com/Adam-TU/remla-project.git

dvc init -f

# add dvc cache remote and link it with json creds (and set as default remote)
dvc remote add -d dvc-cache-remote gdrive://1pwqW-DruetPFaUBeO2KnnnPwccOZGdZw
dvc remote modify dvc-cache-remote gdrive_use_service_account true
dvc remote modify dvc-cache-remote --local gdrive_service_account_json_file_path $KEY_FILE

dvc pull

mv -f models ..

cd ..

gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 src.$APP_MODULE:app