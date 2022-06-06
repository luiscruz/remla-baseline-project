"""Flask API of the Stack Overflow tag prediction model."""
import joblib
from flask import Flask, jsonify, request, Response

# from flasgger import Swagger
from preprocess import text_prepare

MODEL = None
TFIDF_VECTORIZER = None
MLB = None
SET_TAGS = None
SORTED_TAGS_DICT = None
EXT_DS = set()
app = Flask(__name__)
# swagger = Swagger(app)


@app.route("/tags", methods=["GET"])
def tags() -> Response:
    """Give a sorted list of the known tags."""
    return jsonify(SORTED_TAGS_DICT)


@app.route("/predict", methods=["POST"])
def prediction() -> Response:
    """Give prediction result for given title."""
    input_data = request.get_json()
    title = input_data.get("title")
    prediction = predict(title)
    res = {"title": title, "result": prediction}
    return jsonify(res)


@app.route("/submit", methods=["POST"])
def submission() -> Response:
    """Process user-submitted tags for a title."""
    input_data = request.get_json()
    title = input_data.get("title")
    user_tags = input_data.get("result")
    classifier_tags = predict(title)

    EXT_DS.add((title, tuple(user_tags)))

    user_tag_set = set(user_tags)
    overlapping_tag_set = user_tag_set & set(classifier_tags)
    unpredicted_tag_set = user_tag_set - overlapping_tag_set
    manual_tag_set = unpredicted_tag_set & SET_TAGS
    new_tag_set = unpredicted_tag_set - manual_tag_set

    shareChosenTags = len(overlapping_tag_set) / len(classifier_tags)
    shareManualTags = len(manual_tag_set) / len(user_tags)
    shareNewTags = len(new_tag_set) / len(user_tags)
    res = {"shareChosenTags": shareChosenTags, "shareManualTags": shareManualTags, "shareNewTags": shareNewTags}
    return jsonify(res)


def predict(title: str) -> list[int]:
    """Predict tags identifiers for the input title."""
    processed_title_tfidf = TFIDF_VECTORIZER.transform([text_prepare(title)])
    prediction = MLB.inverse_transform(MODEL.predict(processed_title_tfidf))[0]
    return [int(id) for id in prediction]


if __name__ == "__main__":
    MODEL = joblib.load("output/model_tfidf.joblib")
    TFIDF_VECTORIZER = joblib.load("output/tfidf_vectorizer.joblib")
    MLB = joblib.load("output/mlb.joblib")
    sorted_tags = joblib.load("output/sorted_tags.joblib")
    SET_TAGS = set(sorted_tags)
    SORTED_TAGS_DICT = {"result": sorted_tags}
    app.run(host="0.0.0.0", port=8080, debug=False)  # nosec
