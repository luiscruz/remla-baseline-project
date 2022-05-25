import pandas as pd
from src.model import preprocessing

def test_read_file():
    train = preprocessing.read_data('./data/train.tsv')
    validation = preprocessing.read_data('data/validation.tsv')
    test = pd.read_csv('data/test.tsv', sep='\t')

    assert train.empty is False
    assert validation.empty is False
    assert test.empty is False

def test_text_prepare():
	examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
				"How to free c++ memory vector<int> * arr?"]
	answers = ["sql server equivalent excels choose function", 
			   "free c++ memory vectorint arr"]
	for ex, ans in zip(examples, answers):
		if preprocessing.text_prepare(ex) != ans:
			return "Wrong answer for the case: '%s'" % ex
	return 'Basic tests are passed.'

