import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score as roc_auc


def print_evaluation_scores(y_val, predicted):
    f = open("../output/accuracies.txt", "w")
    f.write(f"Accuracy score: {accuracy_score(y_val, predicted)} \n"
            f"F1 score: {f1_score(y_val, predicted, average='weighted')} \n"
            f"Average precision score: {average_precision_score(y_val, predicted, average='macro')}")
    f.close()


def main():
    classifier_mybag, classifier_tfidf = joblib.load("../output/classifiers.joblib")
    X_train_mybag, X_train_tfidf, X_val_mybag, X_val_tfidf = joblib.load("../output/vectorized_x.joblib")
    mlb = joblib.load("../output/mlb.joblib")
    _, X_val, _ = joblib.load("../output/X_preprocessed.joblib")
    y_val =joblib.load("../output/fitted_y_val.joblib")

    # Now you can create predictions for the data. You will need two types of predictions: labels and scores.
    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    # Now take a look at how classifier, which uses TF-IDF, works for a few examples:
    y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
    y_val_inversed = mlb.inverse_transform(y_val)
    for i in range(3):
        print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
            X_val[i],
            ','.join(y_val_inversed[i]),
            ','.join(y_val_pred_inversed[i])
        ))

    print('Bag-of-words')
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
    print('Tfidf')
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

    roc_auc(y_val, y_val_predicted_scores_mybag, multi_class='ovo')
    roc_auc(y_val, y_val_predicted_scores_tfidf, multi_class='ovo')


if __name__ == '__main__':
    main()
