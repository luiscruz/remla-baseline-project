cd /root/dvc-versioning

# TODO: Test this

# checkout a commit in branch dvc-versioning and only get dvc.lock (DO NOT CHECKOUT OHTER BRANCHES)
# the commit must contain valid dvc.lock file 
git checkout $CHECKOUT_COMMIT_HASH -- dvc.lock

# this goes through the dvc.yaml pipeline and checks which version need to be reproduced/fetched
# to get the same version as in dvc.lock for every stage
dvc pull

cp -r models ..