import pickle

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from config.definitions import ROOT_DIR


def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    
    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf


if __name__ == '__main__':
	with open(ROOT_DIR / 'data/processed/train', 'rb') as f:
		X_train_tfidf, y_train = pickle.load(f)

	# Dictionary of all tags from train corpus with their counts.
	tags_counts = {}

	for tags in y_train:
		for tag in tags:
			if tag in tags_counts:
				tags_counts[tag] += 1
			else:
				tags_counts[tag] = 1

	mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
	y_train = mlb.fit_transform(y_train)
	
	classifier_tfidf = train_classifier(X_train_tfidf, y_train)
	
	with open(ROOT_DIR / 'models/tfidf', 'wb') as f:
		pickle.dump(classifier_tfidf, f) # save trained model
	



