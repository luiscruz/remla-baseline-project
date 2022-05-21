#!/bin/bash

#FLASK_ENV=development FLASK_APP=server PYTHONPATH='.:src' flask run
PYTHONPATH='.:src' gunicorn -b 0.0.0.0:8000 server:app
