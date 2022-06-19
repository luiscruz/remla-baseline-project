#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Hielke, Mark
"""

import os

import uvicorn
from fastapi import FastAPI

from .models.predict_model import Evaluator
from .project_types import ModelName

app = FastAPI()


@app.get("/")
def root():
    return "<p>Hello, World!</p>"


@app.get("/metrics/")
def all_metrics():
    evaluator = Evaluator()
    res = {}
    for model_name in [ModelName.bow, ModelName.tfidf]:
        res[model_name] = evaluator.evaluate(model_name)
    return res


@app.get("/metrics/{model_name}")
def metrics(model_name: ModelName):
    evaluator = Evaluator()
    return {model_name: evaluator.evaluate(model_name)}


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
