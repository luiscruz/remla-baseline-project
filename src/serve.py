"""Flask API of the Stack Overflow tag prediction model."""
import joblib
from flask import Flask, jsonify, request

# from flasgger import Swagger
from preprocess import text_prepare

MODEL = None
TFIDF_VECTORIZER = None
MLB = None
SORTED_TAGS = None
app = Flask(__name__)
# swagger = Swagger(app)


@app.route("/tags", methods=["GET"])
def tags():
    """Give a sorted list of the known tags."""
    return jsonify(SORTED_TAGS)


@app.route("/predict", methods=["POST"])
def predict():
    """Predict the tags for a given question."""
    input_data = request.get_json()
    title = input_data.get("title")
    processed_title_tfidf = TFIDF_VECTORIZER.transform([text_prepare(title)])
    prediction = MLB.inverse_transform(MODEL.predict(processed_title_tfidf))[0]
    prediction = [int(id) for id in prediction]
    res = {
        "result": prediction,
        # "classifier": "",
        "title": title,
    }
    return jsonify(res)


if __name__ == "__main__":
    MODEL = joblib.load("output/model_tfidf.joblib")
    TFIDF_VECTORIZER = joblib.load("output/tfidf_vectorizer.joblib")
    MLB = joblib.load("output/mlb.joblib")
    SORTED_TAGS = joblib.load("output/sorted_tags.joblib")
    app.run(host="0.0.0.0", port=8080, debug=True)  # nosec
