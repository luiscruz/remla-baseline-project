import sys

import data
import models
import features
import backend

if __name__ == "__main__":
    if (sys.argv[1] == "--preprocess"):
        features.preprocess_data(input_dir="./data/raw", output_dir="./data/processed")
    elif (sys.argv[1] == "--train"):
        models.start_training()
    elif (sys.argv[1] == "--eval"):
        models.evaluate()
    elif (sys.argv[1] == "--serve"):
        backend.serve()
    else:
        print("Command not found")