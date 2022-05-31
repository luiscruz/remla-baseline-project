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
    print('running...')
    app.run(port=8080, debug=True)
