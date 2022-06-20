
cd $DVC_VERSIONING_PATH

git add dvc.lock
git commit -m "Training Finalized"
git push

dvc push