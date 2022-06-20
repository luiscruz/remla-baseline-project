#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Hielke, Mark
"""

import os
import pickle
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.data.make_dataset import text_prepare
from src.features.build_features import FeatureExtractors
from src.models.predict_model import Evaluator
from src.project_types import ModelName

source_file = Path(__file__)
project_dir = source_file.parent.parent


app = FastAPI()


class Text(BaseModel):
    data: str


@app.get("/")
async def root():
    return HTMLResponse(content="<p>Hello, World!</p>")


@app.get("/metrics/")
async def all_metrics():
    evaluator = Evaluator()
    return {
        model_name: evaluator.evaluate(model_name)
        for model_name in [ModelName.bow, ModelName.tfidf]
    }


@app.get("/metrics/{model_name}")
async def metrics(model_name: ModelName):
    evaluator = Evaluator()
    return {model_name: evaluator.evaluate(model_name)}


@app.post("/predict/{model_name}")
async def predict(model_name: ModelName, text: Text):
    val = text_prepare(text.data)

    processed_dir = project_dir.joinpath("data/processed")
    model_dir = project_dir / "models/"

    X_train = pickle.load(processed_dir.joinpath("X_train.pickle").open("rb"))
    mlb = pickle.load(processed_dir.joinpath("mlb.pickle").open("rb"))

    feature_extractor = FeatureExtractors[model_name](X_train)
    model = pickle.load(model_dir.joinpath(f"{model_name}_model.pickle").open("rb"))

    feature_vector = feature_extractor.get_features([val])  # type: ignore
    predicted_vector = model.predict(feature_vector)
    tags = mlb.inverse_transform(predicted_vector)

    return {"tags": tags}


@app.post("/upload/{date_string}", status_code=status.HTTP_201_CREATED)
async def upload(date_string: str, file: UploadFile):
    contents = file.file.read()
    output_file = project_dir.joinpath(f"data/raw/{date_string}.tsv")
    with output_file.open("wb") as f:
        f.write(contents)
    return


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running FastAPI app on port {port}")

    uvicorn.run(
        "serve_model:app",
        host="0.0.0.0",
        port=port,
        debug=True,
        log_level="debug",
    )
