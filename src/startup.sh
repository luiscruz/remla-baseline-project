#!/usr/bin/env sh
rm -rf $PROMETHEUS_MULTIPROC_DIR
mkdir $PROMETHEUS_MULTIPROC_DIR

dvc init --no-scm

# must contain dvc.lock file and dvc-cache folder
echo "Env. Var. DVC_SHARED_ROOT_PROJECT: $DVC_SHARED_ROOT_PROJECT" 
echo "Current dir: $(pwd)"

sleep 20

# copy dvc-cache folder into root folder of this project
echo "Copying $DVC_SHARED_ROOT_PROJECT/dvc-cache/ into ../"
cp -rl $DVC_SHARED_ROOT_PROJECT/dvc-cache/ ../ 

echo "Copying $DVC_SHARED_ROOT_PROJECT/dvc.lock into ../"
cp $DVC_SHARED_ROOT_PROJECT/dvc.lock ../ 

echo "Setting dvc cache dir to ../dvc-cache"

# assign cache dir to the copied local cache
# this is done so that when dvc-cache is updated using dvc repro
# the dvc-cache in shared root project folder is not modified
# since only continous training service should do that
# inference service only fetches the info of the cache and data and reproduces them
dvc cache dir ../dvc-cache

echo "dvc cache dir set to ../dvc-cache"

# dvc config cache.shared group
# dvc config cache.type symlink

# reproduce the models and output data
dvc repro

gunicorn -c src/gunicorn_config.py -b 0.0.0.0:5000 src.serve_model:app