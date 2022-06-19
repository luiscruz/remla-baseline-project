#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Hielke, Mark
"""

import os
import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello_world():
    return "<p>Hello, World!</p>"


# TODO: Add a predict endpoint...

@app.get("/metrics")
def metrics():
    text = "# HELP my_random A random number\n"
    text += "# TYPE my_random gauge\n"
    text += f"my_random {random.random()}\n\n"

    return Response(text, mimetype='text/plain')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running Flask app on port {port}")

    uvicorn.run("serve_model:app", host="0.0.0.0", port=port, debug=True, port=port, log_level='debug')
