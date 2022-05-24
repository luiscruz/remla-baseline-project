import pandas as pd

from src.model import preprocessing, corpus_counts, bag_of_words, tf_idf, mlb, evaluation

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
	words_counts, tags_counts = corpus_counts.get_corpus_counts(X_train, y_train)
	print(sorted(words_counts, key=words_counts.get, reverse=True)[:3])

	X_train_mybag, X_val_mybag = bag_of_words.initialize(words_counts, X_train, X_val, X_test)

	X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tf_idf.tfidf_features(X_train, X_val, X_test)
	tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}

	mlb_classifier, y_train, y_val = mlb.get_mlb(tags_counts, y_train, y_val)
	classifier_mybag = mlb.train_classifier(X_train_mybag, y_train)
	classifier_tfidf = mlb.train_classifier(X_train_tfidf, y_train)

	y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
	y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)

	print('Bag-of-words')
	evaluation.print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
	print('Tfidf')
	evaluation.print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

	evaluation.print_words_for_tag(classifier_tfidf, 'c', mlb_classifier.classes, tfidf_reversed_vocab, bag_of_words.ALL_WORDS)
	evaluation.print_words_for_tag(classifier_tfidf, 'c++', mlb_classifier.classes, tfidf_reversed_vocab, bag_of_words.ALL_WORDS)
	evaluation.print_words_for_tag(classifier_tfidf, 'linux', mlb_classifier.classes, tfidf_reversed_vocab, bag_of_words.ALL_WORDS)


if __name__ == "__main__":
	main()

