#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Hielke, Mark
"""

import os
import random

from flask import Flask, Response

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


# TODO: Add a predict endpoint...


@app.route("/metrics")
def metrics():
    text = "# HELP my_random A random number\n"
    text += "# TYPE my_random gauge\n"
    text += f"my_random {random.random()}\n\n"

    return Response(text, mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
