import shutil
import os

DIRECTORY_FILES = "src"
DIRECTORY_COPY = "tests/dependencies"


for file_name in os.listdir(DIRECTORY_FILES):

    file_location = DIRECTORY_FILES+"/"+file_name
    new_file_location = DIRECTORY_COPY+"/"+file_name.split("_")[-1]

    shutil.copyfile(file_location, new_file_location)
