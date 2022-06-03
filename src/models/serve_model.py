"""
Flask API of the SMS Spam detection model model.
"""
<<<<<<< HEAD
import pickle
from flask import Flask, jsonify, request
from flasgger import Swagger
=======
import traceback
import pickle
from flask import Flask, jsonify, request, Response
from flasgger import Swagger
import pandas as pd
import random
>>>>>>> 2cd4689bc757856a3c45a594054eb4c483a06b93

from src.config.definitions import ROOT_DIR
from src.features.build_features import text_prepare

app = Flask(__name__)
swagger = Swagger(app)

<<<<<<< HEAD
=======
num_pred = 0
>>>>>>> 2cd4689bc757856a3c45a594054eb4c483a06b93

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the tag belonging to the title of a post of StackOverflow.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: title to predict tags for.
          required: True
          schema:
            type: object
            required: title
            properties:
                title:
                    type: string
                    example: This is an example of a title.
    responses:
      200:
<<<<<<< HEAD
        description: "The result of the prediction, a list of tags (e.g. 'python', 'c++' and/or 'javascript')"
=======
        description: "The result of the prediction (e.g. 'python', 'c++' or 'javascript')"
>>>>>>> 2cd4689bc757856a3c45a594054eb4c483a06b93
    """
    input_data = request.get_json()
    title = input_data.get('title')

    prepared_title = text_prepare(title) # remove bad symbols
    processed_title = tfidf_vectorizer.transform([prepared_title])

    with open(ROOT_DIR / 'models/tfidf.pkl', 'rb') as f:
<<<<<<< HEAD
        model = pickle.load(f)
    prediction = model.predict(processed_title)[0]

    return jsonify({
        "result": prediction.tolist(),
        "classifier": "tfifd multi-label-binarizer",
=======
    	model = pickle.load(f)
    prediction = model.predict(processed_title)[0]

    global num_pred
    num_pred = num_pred + 1  # Increment number of total predictions made

    # TODO: Convert preediction: binary array -> list of tags as strings ?
    
    return jsonify({
        "result": prediction.tolist(),
        "classifier": "tfifd multi-label-binarizer ",
>>>>>>> 2cd4689bc757856a3c45a594054eb4c483a06b93
        "title": title
    })


<<<<<<< HEAD
if __name__ == '__main__':
    with open(ROOT_DIR / 'data/derivates/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    app.run(port=8080, debug=True)
=======
@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Get metrics for monitoring.
    """

    string = ""
    string += "# HELP my_random A random number\n"
    string += "# TYPE my_random gauge\n"
    string += "my_random " + str(random.randint(0,100)) + "\n\n"

    string += "# HELP num_pred Number of total predictions made\n"
    string += "# TYPE num_pred counter\n"
    string += "num_pred " + str(num_pred) + "\n\n"

    # Note: Prometheus requires mimetype to be explicitly set to text/plain
    return Response(string, mimetype='text/plain')


if __name__ == '__main__':
    with open(ROOT_DIR / 'data/derivates/tfidf_vectorizer.pkl', 'rb') as f:
    	tfidf_vectorizer = pickle.load(f)
    app.run(host='0.0.0.0', port=8080, debug=True)
>>>>>>> 2cd4689bc757856a3c45a594054eb4c483a06b93
