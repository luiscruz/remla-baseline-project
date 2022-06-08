#!/usr/bin/env sh
rm -rf $PROMETHEUS_MULTIPROC_DIR
mkdir $PROMETHEUS_MULTIPROC_DIR

dvc init --no-scm

echo $DVC_CACHE_PATH

sleep 20

dvc cache dir $DVC_CACHE_PATH
dvc config cache.shared group
dvc config cache.type symlink

dvc pull -f

gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 src.serve_model:app