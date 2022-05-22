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
	"""
		text: a string
		
		return: modified initial string
	"""
	text = text.lower() # lowercase text
	text = re.sub(REPLACE_BY_SPACE_RE, " ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text
	text = re.sub(BAD_SYMBOLS_RE, "", text) # delete symbols which are in BAD_SYMBOLS_RE from text
	text = " ".join([word for word in text.split() if not word in STOPWORDS]) # delete stopwords from text
	return text

def test_text_prepare():
	examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
				"How to free c++ memory vector<int> * arr?"]
	answers = ["sql server equivalent excels choose function", 
			   "free c++ memory vectorint arr"]
	for ex, ans in zip(examples, answers):
		if text_prepare(ex) != ans:
			return "Wrong answer for the case: '%s'" % ex
	return 'Basic tests are passed.'
