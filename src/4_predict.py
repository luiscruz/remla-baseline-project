from joblib import dump, load


def main():

    # Load the model and the X_val to use
    classifier = load('output/classifier.joblib')
    X_val = load('output/bag_of_words.joblib')["X_val"]

    # This one would be cool that it reads the arguments for the validation and do the rest.

    print(X_val)
    print("HHAS")
    labels = classifier.predict(X_val)
    scores = classifier.decision_function(X_val)

    print(labels)
    print(scores)

    results = {"labels": labels,
               "scores": scores}

    dump(results, 'output/prediction_results.joblib')


if __name__ == "__main__":
    main()
