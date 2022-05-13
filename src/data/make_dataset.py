import os
import nltk
import sys


from config.definitions import ROOT_DIR

if __name__ == '__main__':
	dirpath = ROOT_DIR / 'data/external'
	nltk.download('stopwords', download_dir=dirpath)
