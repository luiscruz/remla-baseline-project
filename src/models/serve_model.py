"""
Flask API of the SMS Spam detection model model.
"""
import traceback
import pickle
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd

from src.config.definitions import ROOT_DIR
from src.features.build_features import text_prepare

app = Flask(__name__)
swagger = Swagger(app)


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
        description: "The result of the prediction (e.g. 'python', 'c++' or 'javascript')"
    """
    input_data = request.get_json()
    title = input_data.get('title')

    prepared_title = text_prepare(title) # remove bad symbols
    processed_title = tfidf_vectorizer.transform([prepared_title])

    with open(ROOT_DIR / 'models/tfidf.pkl', 'rb') as f:
    	model = pickle.load(f)
    prediction = model.predict(processed_title)[0]

    # TODO: Convert preediction: binary array -> list of tags as strings ?
    
    return jsonify({
        "result": prediction.tolist(),
        "classifier": "tfifd multi-label-binarizer ",
        "title": title
    })


if __name__ == '__main__':
    with open(ROOT_DIR / 'data/derivates/tfidf_vectorizer.pkl', 'rb') as f:
    	tfidf_vectorizer = pickle.load(f)
    app.run(port=8080, debug=True)
