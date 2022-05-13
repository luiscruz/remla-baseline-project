from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from joblib import dump, load


def print_evaluation_scores(y_val, predicted, prediction_results):
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))
    print(f'Roc result: {roc_auc_score(y_val, prediction_results["scores"], multi_class="ovo")}')


# Change variable names
def main():
    prediction_results = load('output/prediction_results.joblib')

    scores = prediction_results["scores"]
    labels = prediction_results["labels"]
    y_val = load('output/val_data.joblib')

    print_evaluation_scores(y_val, labels, prediction_results)


if __name__ == "__main__":
    main()
