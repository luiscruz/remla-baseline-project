import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from src import preprocessing, tag_count, bag_of_words, tf_idf, mlb, evaluation

def main():

	train = preprocessing.read_data('./data/train.tsv')
	validation = preprocessing.read_data('data/validation.tsv')
	test = pd.read_csv('data/test.tsv', sep='\t')

	X_train, y_train = train['title'].values, train['tags'].values
	X_val, y_val = validation['title'].values, validation['tags'].values
	X_test = test['title'].values

	print(train.head())
	print(preprocessing.test_text_prepare())

	X_train = [preprocessing.text_prepare(x) for x in X_train]
	X_val = [preprocessing.text_prepare(x) for x in X_val]
	X_test = [preprocessing.text_prepare(x) for x in X_test]

	# Dictionary of all words from train corpus with their counts.
	words_counts, tags_counts = tag_count.get_tag_count(X_train, y_train)
	print(sorted(words_counts, key=words_counts.get, reverse=True)[:3])

	print(bag_of_words.test_my_bag_of_words())
	X_train_mybag, X_val_mybag = bag_of_words.initialize(words_counts, X_train, X_val, X_test)

	X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tf_idf.tfidf_features(X_train, X_val, X_test)
	tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}

	mlb_classifier, y_train, y_val = mlb.get_mlb(tags_counts, y_train, y_val)
	classifier_mybag = train_classifier(X_train_mybag, y_train)
	classifier_tfidf = train_classifier(X_train_tfidf, y_train)

	y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
	y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

	y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
	y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

	# y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
	# y_val_inversed = mlb.inverse_transform(y_val)
	# for i in range(3):
	# 	print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
	# 		X_val[i],
	# 		','.join(y_val_inversed[i]),
	# 		','.join(y_val_pred_inversed[i])
	# 	))

	print('Bag-of-words')
	evaluation.print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
	print('Tfidf')
	evaluation.print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

	print_words_for_tag(classifier_tfidf, 'c', mlb_classifier.classes, tfidf_reversed_vocab, bag_of_words.ALL_WORDS)
	print_words_for_tag(classifier_tfidf, 'c++', mlb_classifier.classes, tfidf_reversed_vocab, bag_of_words.ALL_WORDS)
	print_words_for_tag(classifier_tfidf, 'linux', mlb_classifier.classes, tfidf_reversed_vocab, bag_of_words.ALL_WORDS)


def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary
        
        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))
    
    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator. 
    
    model = classifier.estimators_[tags_classes.index(tag)]
    top_positive_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][-5:]]
    top_negative_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][:5]]
    
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))

def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create a nd fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C,
                             dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf


if __name__ == "__main__":
	main()

