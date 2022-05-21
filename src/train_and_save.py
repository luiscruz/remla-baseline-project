from persistence import train_and_save_models
from argparse import ArgumentParser


DEFAULT_EXPORTS_PATH = '../exports/model.joblib'

parser = ArgumentParser()
parser.add_argument('--model-path', type=str, default='model.joblib')
parser.add_argument('--data-dir', type=str, default='./data-dir')
args = parser.parse_args()

train_and_save_models(args.data_dir, args.model_path)
