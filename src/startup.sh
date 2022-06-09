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

# this is inefficient, but we need the files to be copied so that any changes to cache does not effect the mounted cache
# dvc config cache.shared group
# dvc config cache.type symlink 
# dvc unprotect $DVC_SHARED_ROOT_PROJECT/dvc-cache/

# dvc config cache.shared group
# dvc config cache.type symlink

# reproduce the models and output data
dvc repro

gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 src.serve_model:app