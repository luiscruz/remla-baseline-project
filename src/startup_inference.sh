# #!/usr/bin/env sh
# rm -rf $PROMETHEUS_MULTIPROC_DIR
# mkdir $PROMETHEUS_MULTIPROC_DIR

# # # load google drive api key secret into file for use by DVC
# # echo $API_KEY_SECRET > remla-352721-99f80e5bc090.json

# # dvc init --no-scm -f

# # # create config entries for gdrive authentication to go automatically 
# # # by utilizing the API_KEY_SECRET json credentials

# # # add dvc cache remote and link it with json creds (and set as default remote)
# # dvc remote add -d dvc-cache-remote gdrive://1pwqW-DruetPFaUBeO2KnnnPwccOZGdZw
# # dvc remote modify dvc-cache-remote gdrive_use_service_account true
# # dvc remote modify dvc-cache-remote --local gdrive_service_account_json_file_path remla-352721-99f80e5bc090.json

# # dvc pull

# gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 src.$APP_MODULE:app