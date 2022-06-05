"""
Flask API of the SMS Spam detection model model.
"""
import os
import pickle
import shutil

import yaml
from flasgger import Swagger
from flask import Flask, Response, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Summary,
    generate_latest,
    multiprocess,
)

from src.preprocess.preprocess_data import text_prepare

PROMETHEUS_MULTIPROC_DIR = os.environ["PROMETHEUS_MULTIPROC_DIR"]
# make sure the dir is clean
shutil.rmtree(PROMETHEUS_MULTIPROC_DIR, ignore_errors=True)
os.makedirs(PROMETHEUS_MULTIPROC_DIR)

app_name = "inference-service"
app = Flask(app_name)
swagger = Swagger(app)

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

duration_metric = Summary("predict_duration", "Time spent per predict request")


def load_yaml_params():
    # Fetch params from yaml params file
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pickle(path_to_pkl):
    with open(path_to_pkl, "rb") as fd:
        return pickle.load(fd)


def init_app():
    params = load_yaml_params()
    train_params = params["train"]
    feature_params = params["featurize"]

    model_path = train_params["model_out"]
    mlb_path = feature_params["mlb_out"]
    vectorizer_path = feature_params["tfidf_vectorizer_out"]  # responsible for featurizing text into tfidf vectors

    global MODEL, MLB, TFIDF_VECTORIZER
    MODEL = load_pickle(model_path)
    MLB = load_pickle(mlb_path)
    TFIDF_VECTORIZER = load_pickle(vectorizer_path)


init_app()


@app.route("/predict", methods=["POST"])
@duration_metric.time()
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
    processed_title = text_prepare(title)

    featurized_title = TFIDF_VECTORIZER.transform([processed_title])
    prediction = MODEL.predict(featurized_title)
    tags = MLB.inverse_transform(prediction)

    res = {"tags": tags, "classifier": "decision tree", "title": title}
    app.logger.debug(f"prediction: {res}")
    return jsonify(res)


@app.route("/metrics")
def metrics():
    data = generate_latest(registry)
    app.logger.debug(f"Metrics, returning: {data}")
    return Response(data, mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)  # nosec
