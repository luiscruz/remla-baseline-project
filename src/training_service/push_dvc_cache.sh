
APP_PATH=/app
GIT_REPO_PATH=$APP_PATH/dvc-versioning

cd $APP_PATH

cp dvc.lock $GIT_REPO_PATH

cd $GIT_REPO_PATH

git add dvc.lock
git commit -m "Training Finalized"
git push

cd $APP_PATH

dvc push