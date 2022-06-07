#!/usr/bin/env sh
dvc cache dir $DVC_CACHE_PATH
dvc config cache.shared group
dvc config cache.type symlink

sleep 60

dvc init --no-scm

dvc repro

gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 src.serve_model:app