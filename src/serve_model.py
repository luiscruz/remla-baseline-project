#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author hielke
"""
import os

from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


# TODO: Add a predict endpoint...


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
