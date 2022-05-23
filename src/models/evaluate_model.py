from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import math
import pickle
import json
import yaml

# Fetch params from yaml params file
params = yaml.safe_load(open("params.yaml"))
train_params = params['train']
featurize_params = params['featurize']
evaulate_params = params['evaluate']

MODEL_PATH = train_params['model_out']
VAL_DATA_PATH = featurize_params['output_val']
# TEST_DATA_SET = featurize_params['output_test']

# PRC_IMG_PATH = evaulate_params['prc_img']
# ROC_IMG_PATH = evaulate_params['roc_img']
PRC_JSON_PATH = evaulate_params['prc_json']
ROC_JSON_PATH = evaulate_params['roc_json'] 
SCORES_JSON_PATH = evaulate_params['scores_path']

def create_all_plots_and_scores(
    classifier_tfidf,
    X_val_tfidf,
    y_val_tdidf
):
    # y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    # y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    create_evaluation_scores_json(
        y_val_tdidf,
        y_val_predicted_labels_tfidf,
        y_val_predicted_scores_tfidf
    )

    # create_prc_curve_json(y_val_tdidf, y_val_predicted_scores_tfidf)
    # create_roc_curve_json(y_val_tdidf, predicted_probas_tfidf)

    # A look at how classifier, which uses TF-IDF, works for a few examples:
    # y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
    # y_val_inversed = mlb.inverse_transform(y_val)
    # for i in range(3):
    #     print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
    #         X_val[i],
    #         ','.join(y_val_inversed[i]),
    #         ','.join(y_val_pred_inversed[i])
    #     ))

    # alternative_preprocessing_models(X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, mlb)


# def print_words_for_tag(classifier, tag, tags_classes, index_to_words):
#     """
#     Look at the features (words or n-grams) that are used with the largest weights in your logistic regression model.
#     classifier: trained classifier
#     tag: particular tag
#     tags_classes: a list of classes names from MultiLabelBinarizer
#     index_to_words: index_to_words transformation
#     all_words: all words in the dictionary

#     :return: nothing, just print top 5 positive and top 5 negative words for current tag
#     """
#     print('Tag:\t{}'.format(tag))

#     # Extract an estimator from the classifier for the given tag.
#     # Extract feature coefficients from the estimator.

#     model = classifier.estimators_[tags_classes.index(tag)]
#     top_positive_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][-5:]]
#     top_negative_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][:5]]

#     print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
#     print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))


"""
Print out scores
"""


def create_evaluation_scores_json(y_val, predicted_labels, decision_function_vals):
    """

    :param y_val:
    :param predicted:
    :param probas:
    :return: Create json file of scores
    """
    print(y_val)
    with open(SCORES_JSON_PATH, "w") as fd:
        json.dump(
            {
                'accuracy_score': accuracy_score(y_val, predicted_labels),
                'f1_score': f1_score(y_val, predicted_labels, average='weighted'),
                'avg_precision_score': average_precision_score(y_val, predicted_labels, average='macro'),
                'roc_auc_score': roc_auc(y_val, decision_function_vals, multi_class='ovo'),
            },
            fd,
            indent=4,
        )
    

# def create_prc_curve_json(y_val, probas_pred):
#     """
#     Create json file of prc points
#     """
#     precision, recall, prc_thresholds = precision_recall_curve(y_val, probas_pred)
#     nth_point = math.ceil(len(prc_thresholds) / 1000)
#     prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
#     with open(PRC_JSON_PATH, "w") as fd:
#         json.dump(
#             {
#                 "prc": [
#                     {"precision": p, "recall": r, "threshold": t}
#                     for p, r, t in prc_points
#                 ]
#             },
#             fd,
#             indent=4,
#         )

# def create_roc_curve_json(y_val, probas_pred):
#     """
#     Create json file of roc points
#     """
#     fpr, tpr, roc_thresholds = roc_curve(y_val, probas_pred)
#     roc_points = list(zip(fpr, tpr, roc_thresholds))
#     with open(ROC_JSON_PATH, "w") as fd:
#         json.dump(
#             {
#                 "roc": [
#                     {"fpr": fpr, "tpr": tpr, "threshold": t}
#                     for fpr, tpr, t in roc_points
#                 ]
#             },
#             fd,
#             indent=4,
#         )

# def print_eval_scores(
#     y_val,
#     # y_val_predicted_labels_mybag,
#     y_val_predicted_labels_tfidf
# ):
#     """

#     :param y_val:
#     :param y_val_predicted_labels_mybag:
#     :param y_val_predicted_labels_tfidf:
#     :return: Nothing, prints the evaluation scores for bag of words and tfidf.
#     """
#     # print('Bag-of-words')
#     # print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
#     print('Tfidf')
#     print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


# def roc_auc_scores(
#     y_val,
#     # y_val_predicted_scores_mybag,
#     y_val_predicted_scores_tfidf
# ):
#     """

#     :param y_val:
#     :param y_val_predicted_scores_mybag:
#     :param y_val_predicted_scores_tfidf:
#     :return: Nothing, prints roc score for bag of words and tfidf.
#     """

#     # mybag_roc = roc_auc(y_val, y_val_predicted_scores_mybag, multi_class='ovo')
#     # print('Bag-of-words: ', mybag_roc)
#     tfidf_roc = roc_auc(y_val, y_val_predicted_scores_tfidf, multi_class='ovo')
#     print('Tfidf: ', tfidf_roc)


"""
Task 4 - MultilabelClassification
"""


# def alternative_preprocessing_models(
#     classifier_tfidf,
#     X_train_tfidf,
#     X_val_tfidf,
#     X_test_tfidf,
#     y_train,
#     mlb
# ):
#     """
#     Once the evaluation is set up, experiment with training your classifiers. We will use *F1-score weighted* as an evaluation metric.
#     Our recommendation:
#     - compare the quality of the bag-of-words and TF-IDF approaches and chose one of them.
#     - for the chosen one, try *L1* and *L2*-regularization techniques in Logistic Regression with different coefficients (e.g. C equal to 0.1, 1, 10, 100).
#     :return:
#     """

#     # coefficients = [0.1, 1, 10, 100]
#     # penalties = ['l1', 'l2']

#     # for coefficient in coefficients:
#     #     for penalty in penalties:
#     #         classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty=penalty, C=coefficient)
#     #         y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
#     #         y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
#     #         print("Coefficient: {}, Penalty: {}".format(coefficient, penalty))
#     #         print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

#     test_predictions = classifier_tfidf.predict(X_test_tfidf)
#     test_pred_inversed = mlb.inverse_transform(test_predictions)

#     test_predictions_for_submission = '\n'.join(
#         '%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))
#     print(test_predictions_for_submission)

def load_model():
    with open(MODEL_PATH, 'rb') as fd:
        return pickle.load(fd)

def load_val_data():
    with open(VAL_DATA_PATH, 'rb') as fd:
        X_val, y_val = pickle.load(fd)
    return X_val, y_val

def main():
    clf = load_model()
    X_val, y_val = load_val_data()
    create_all_plots_and_scores(clf, X_val, y_val)

if __name__ == '__main__':
    main()
