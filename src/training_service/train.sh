# TODO: check whether a simple cp command to copy the train,val and test is enough
# instead of dvc adding them each time training needs to be done
rm -rf /app/data/

mkdir -p /app/data/processed
mkdir -p /app/data/interim
mkdir -p /app/data/external

dvc add $SHARED_DATA_PATH/raw/ -o /app/data/

# cd to /app first since dvc.yaml is only present there and dvc repro will not work without it
cd /app && dvc repro -q