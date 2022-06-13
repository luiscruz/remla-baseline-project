# TODO: check whether a simple cp command to copy the train,val and test is enough
# instead of dvc adding them each time training needs to be done
rm -rf data/

# mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/interim
mkdir -p data/external

dvc add $SHARED_DATA_PATH/raw/ -o data/
# dvc add $SHARED_DATA_PATH/raw/test.tsv -o data/raw/test.tsv
# dvc add $SHARED_DATA_PATH/raw/train.tsv -o data/raw/train.tsv
# dvc add $SHARED_DATA_PATH/raw/validation.tsv -o data/raw/validation.tsv

dvc repro -q