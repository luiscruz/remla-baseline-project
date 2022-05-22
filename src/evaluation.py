
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

def print_evaluation_scores(y_val, predicted):
	print('Accuracy score: ', accuracy_score(y_val, predicted))
	print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
	print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))
