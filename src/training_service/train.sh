# TODO: check whether a simple cp command to copy the train,val and test is enough
# instead of dvc adding them each time training needs to be done
APP_PATH=/app

rm -rf $APP_PATH/data/

mkdir -p $APP_PATH/data/processed
mkdir -p $APP_PATH/data/interim
mkdir -p $APP_PATH/data/external

dvc add $SHARED_DATA_PATH/raw/ -o $APP_PATH/data/

# cd to /app first since dvc.yaml is only present there and dvc repro will not work without it
cd $APP_PATH && dvc repro -q