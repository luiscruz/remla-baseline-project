#!/usr/bin/env sh
rm -rf $PROMETHEUS_MULTIPROC_DIR
mkdir $PROMETHEUS_MULTIPROC_DIR
gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 src.serve_model:app