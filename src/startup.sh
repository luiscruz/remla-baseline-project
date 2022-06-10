#!/usr/bin/env sh
rm -rf $PROMETHEUS_MULTIPROC_DIR
mkdir $PROMETHEUS_MULTIPROC_DIR

dvc init --no-scm

# must contain dvc.lock file and dvc-cache folder
echo "Env. Var. DVC_CACHE_PATH: $DVC_CACHE_PATH" 
echo "Current dir: $(pwd)"

# add sleep if dvc-cache in host is empty since cp won't work
# if there is no cache to rebuild
# sleep 35

# copy dvc-cache folder into root folder of this project
echo "Copying $DVC_CACHE_PATH into current pwd"
cp -r $DVC_CACHE_PATH .

echo "Setting dvc cache dir to dvc-cache"

# assign cache dir to the copied local cache
# this is done so that when dvc-cache is updated using dvc repro
# the dvc-cache in shared root project folder is not modified
# since only continous training service should do that
# inference service only fetches the info of the cache and data and reproduces them
dvc cache dir dvc-cache

echo "Cache dir set to dvc-cache"
 
# For DVC ADD command so that it fetches the raw data from shared storage
# NOTE: this folder must be removed and re-created when the data needs to be fetched again.
mkdir -p data/raw

# recreating other required folders for dvc repro to succeed
mkdir -p data/processed
mkdir -p data/interim
mkdir -p data/external

# fetching the input data for dvc repro from shared folder
dvc add $SHARED_DATA_PATH/raw/test.tsv -o data/raw/test.tsv
dvc add $SHARED_DATA_PATH/raw/text_prepare_tests.tsv -o data/raw/text_prepare_tests.tsv
dvc add $SHARED_DATA_PATH/raw/train.tsv -o data/raw/train.tsv
dvc add $SHARED_DATA_PATH/raw/validation.tsv -o data/raw/validation.tsv

# reproduce the models and output data
dvc repro

#TODO: uncomment this again after testing
# gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 src.serve_model:app
sleep 2312323