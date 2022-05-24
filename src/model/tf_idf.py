from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_val, X_test):
	"""
		X_train, X_val, X_test — samples        
		return TF-IDF vectorized representation of each sample and vocabulary
	"""
	# Create TF-IDF vectorizer with a proper parameters choice
	# Fit the vectorizer on the train set
	# Transform the train, test, and val sets and return the result
	
	
	tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1,2), token_pattern='(\S+)') ####### YOUR CODE HERE #######
	
	X_train = tfidf_vectorizer.fit_transform(X_train)
	X_val = tfidf_vectorizer.transform(X_val)
	X_test = tfidf_vectorizer.transform(X_test)
	
	return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_