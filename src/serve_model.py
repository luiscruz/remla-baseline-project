#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Hielke, Mark
"""

import os
import pickle
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from .features.build_features import FeatureExtractorBow, FeatureExtractorTfidf
from .models.predict_model import Evaluator
from .project_types import ModelName
from .util.util import read_data

source_file = Path(__file__)
project_dir = source_file.parent.parent.parent


app = FastAPI()

class Text(BaseModel):
    data: str



@app.get("/")
async def root():
    return "<p>Hello, World!</p>"


@app.get("/metrics/")
async def all_metrics():
    evaluator = Evaluator()
    res = {}
    for model_name in [ModelName.bow, ModelName.tfidf]:
        res[model_name] = evaluator.evaluate(model_name)
    return res


@app.get("/metrics/{model_name}")
async def metrics(model_name: ModelName):
    evaluator = Evaluator()
    return {model_name: evaluator.evaluate(model_name)}


@app.post("/predict/{model_name}")
async def predict(model_name: ModelName, text: Text):
    val = text['data']

    processed_dir = project_dir.joinpath("data/processed")
    model_dir = project_dir / 'models/'

    X_train = pickle.load(processed_dir.joinpath("X_train.pickle").open('rb'))
    mlb = pickle.load(processed_dir.joinpath("mlb.pickle").open('rb'))

    if model_name == ModelName.tfidf:
        feature_extractor = FeatureExtractorTfidf(X_train)
    elif model_name == ModelName.bow:
        feature_extractor = FeatureExtractorBow(X_train)

    model = pickle.load(model_dir.joinpath(f"{model}_mode.pickle").open('rb'))

    feature_vector = feature_extractor.get_features()
    predicted_vector = model.predict(feature_vector)
    tags = mlb.inverse_transform(predicted_vector)

    return {
        "tags": tags
    }


@app.post("/upload/")
def upload(datum_string: str):

    file ...



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running FastAPI app on port {port}")

    uvicorn.run(
        "serve_model:app",
        host="0.0.0.0",
        port=port,
        debug=True,
        port=port,
        log_level="debug",
    )
