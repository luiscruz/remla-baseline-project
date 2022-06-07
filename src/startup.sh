#!/usr/bin/env sh
rm -rf $PROMETHEUS_MULTIPROC_DIR
mkdir $PROMETHEUS_MULTIPROC_DIR

dvc cache dir $DVC_CACHE_PATH
dvc config cache.shared group
dvc config cache.type symlink

sleep 120

echo "Slept for 120 seconds"
dvc pull

gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 src.serve_model:app