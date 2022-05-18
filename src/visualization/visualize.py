from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score as roc_auc

def print_evaluation_scores(y_val, predicted):
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))

def print_eval_scores(y_val, y_val_predicted_labels_mybag, y_val_predicted_labels_tfidf):
    print('Bag-of-words')
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
    print('Tfidf')
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

def roc_auc_scores(y_val, y_val_predicted_scores_mybag, y_val_predicted_scores_tfidf):
    roc_auc(y_val, y_val_predicted_scores_mybag, multi_class='ovo')
    roc_auc(y_val, y_val_predicted_scores_tfidf, multi_class='ovo')
