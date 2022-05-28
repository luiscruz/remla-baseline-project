"""
Flask API of the SMS Spam detection model model.
"""
import pickle

import yaml
from flasgger import Swagger

# import traceback
from flask import Flask, jsonify, request

from src.preprocess.preprocess_data import text_prepare

app_name = "inference-service"
app = Flask(app_name)
swagger = Swagger(app)


def load_yaml_params():
    # Fetch params from yaml params file
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pickle(path_to_pkl):
    with open(path_to_pkl, "rb") as fd:
        return pickle.load(fd)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether an SMS is Spam.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: list of tags as strings."
    """
    input_data = request.get_json()
    title = input_data.get("title")
    # processed_title = text_prepare(title)
    tags = "this is a list of tags".split(" ")
    res = {"tags": tags, "classifier": "decision tree", "title": title}
    print(res)
    return jsonify(res)
