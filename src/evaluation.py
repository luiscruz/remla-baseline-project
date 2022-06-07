"""
    Main logic of this stage:
    * loads the loblibs
    * evaluates the models
    * prints results in ../output/accuracies.txt
"""
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

output_directory = "output"


def calculate_evaluation_scores(model_name, y_val, predicted):
    """
        y_val: ground truth labels
        predicted: predicted labels

        Calculate the evaluation results and save results in joblib and an output file
    """
    # Calculate scores
    accuracy = accuracy_score(y_val, predicted)
    f1 = f1_score(y_val, predicted, average='weighted')
    avg_precision = average_precision_score(y_val, predicted, average='macro')
    # Write to joblib
    joblib.dump((accuracy, f1, avg_precision), output_directory + f"/{model_name}_scores.joblib")
    # Write to output file
    with open(output_directory + f"/{model_name}_accuracies.txt", "w", encoding='utf-8') as f:
        f.write(f"Accuracy score: {accuracy} \n"
                f"F1 score: {f1} \n"
                f"Average precision score: {avg_precision}")
        f.close()


def main():
    """
        Main logic of this stage:
        * loads the loblibs
        * evaluates the models
        * prints results in ../output/accuracies.txt
    """
    classifier_mybag, classifier_tfidf = joblib.load(output_directory + "/classifiers.joblib")
    _, _, X_val_mybag, X_val_tfidf = joblib.load(output_directory + "/vectorized_x.joblib")
    mlb = joblib.load(output_directory + "/mlb.joblib")
    _, X_val, _ = joblib.load(output_directory + "/X_preprocessed.joblib")
    _, y_val =joblib.load(output_directory + "/fitted_y.joblib")


    # Now you can create predictions for the data. You will need two types of predictions: labels and scores.
    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    # Now take a look at how classifier, which uses TF-IDF, works for a few examples:
    y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
    y_val_inversed = mlb.inverse_transform(y_val)
    for i in range(3):
        print(f"Title:\t{X_val[i]}\n"
              f"True labels:\t{','.join(y_val_inversed[i])}\n"
              f"Predicted labels:\t{','.join(y_val_pred_inversed[i])}\n\n")

    print('Bag-of-words')
    calculate_evaluation_scores("BOW", y_val, y_val_predicted_labels_mybag)
    print('Tfidf')
    calculate_evaluation_scores("TFIDF", y_val, y_val_predicted_labels_tfidf)

    roc_auc_score(y_val, y_val_predicted_scores_mybag, multi_class='ovo')
    roc_auc_score(y_val, y_val_predicted_scores_tfidf, multi_class='ovo')


if __name__ == '__main__':
    main()
