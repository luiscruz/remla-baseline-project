#!/usr/bin/env sh

dvc init --no-scm

echo $DVC_CACHE_PATH

dvc cache dir $DVC_CACHE_PATH
dvc config cache.shared group
dvc config cache.type symlink

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

# reproduce pipeline to create the models and output data
# this also updates the cache if something has changed in the data
dvc repro

#TODO: uncomment this again after testing
# gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 src.serve_model:app
sleep 2312323