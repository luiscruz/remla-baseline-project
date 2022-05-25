import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from ast import literal_eval

nltk.download('stopwords')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def read_data(filename):
	data = pd.read_csv(filename, sep='\t')
	data['tags'] = data['tags'].apply(literal_eval)
	return data

def text_prepare(text):
	text = text.lower()
	text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
	text = re.sub(BAD_SYMBOLS_RE, "", text)
	text = " ".join([word for word in text.split() if not word in STOPWORDS])
	return text
