from joblib import dump, load


def run_prediction(classifier, X_val):
    labels = classifier.predict(X_val)
    scores = classifier.decision_function(X_val)

    return {"labels": labels, "scores": scores}


def main():

    classifiers = load('output/classifiers.joblib')
    text_process_data = load('output/text_processor_data.joblib')

    predictions = {}
    # Load the model and the X_val to use
    classifier_bag = classifiers['bag']
    X_val_bag = text_process_data["bag"]["X_val"]

    predictions["bag"] = run_prediction(classifier_bag, X_val_bag)

    classifier_tfidf = classifiers['tfidf']
    X_val_tfdif = text_process_data["tfidf"]["X_val"]
    predictions["tfidf"] = run_prediction(classifier_tfidf, X_val_tfdif)
    dump(predictions, 'output/predictions.joblib')


if __name__ == "__main__":
    main()
