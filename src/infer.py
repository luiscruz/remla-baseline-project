from text_preprocessing import process_for_inference


def infer(data, classifier_mybag, classifier_tfidf, mlb, tfidf_vectorizer, words_to_index, dict_size):
    X_mybag, X_tfidf = process_for_inference(data, words_to_index, dict_size, tfidf_vectorizer)

    y_mybag_enc = classifier_mybag.predict(X_mybag)
    y_tfidf_enc = classifier_tfidf.predict(X_tfidf)

    y_mybag = mlb.inverse_transform(y_mybag_enc)
    y_tfidf = mlb.inverse_transform(y_tfidf_enc)
    # print(y_mybag)
    # print(y_tfidf)
    return y_mybag, y_tfidf
