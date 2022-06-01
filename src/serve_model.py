#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author hielke
"""
from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# TODO: Add a predict endpoint...

if __name__ == '__main__':
    PORT = 8080
    print(f'Running Flask app on port {PORT}')
    app.run(port=PORT, debug=True)
