import shutil
import os


def copy_file_import(sourcePath):

    # call copyfile() method
    destinationPath = "tests/testing_file.py"
    result = shutil.copyfile(sourcePath, destinationPath)

    if result == destinationPath:
        return True
    else:
        exit(1)


def delete_import():
    os.remove("tests/testing_file.py")
