# TODO: check whether a simple cp command to copy the train,val and test is enough
# instead of dvc adding them each time training needs to be done
rm -rf $DVC_VERSIONING_PATH/data/

mkdir -p $DVC_VERSIONING_PATH/data/processed
mkdir -p $DVC_VERSIONING_PATH/data/interim
mkdir -p $DVC_VERSIONING_PATH/data/external

cd $DVC_VERSIONING_PATH

dvc add $SHARED_DATA_PATH/raw/ -o $DVC_VERSIONING_PATH/data/

# cd to /app first since dvc.yaml is only present there and dvc repro will not work without it
dvc repro -q