import os
import shutil

import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Summary,
    generate_latest,
    multiprocess,
)

PROMETHEUS_MULTIPROC_DIR = os.environ["PROMETHEUS_MULTIPROC_DIR"]
# make sure the dir is clean
shutil.rmtree(PROMETHEUS_MULTIPROC_DIR, ignore_errors=True)
os.makedirs(PROMETHEUS_MULTIPROC_DIR)

app_name = "training-service"
app = Flask(app_name)

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

duration_metric = Summary("train_duration", "Time spent on training")


def load_yaml_params():
    # Fetch params from yaml params file
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.route("/train", methods=["POST"])
@duration_metric.time()
def train():
    print("TRAINING")


@app.route("/metrics")
def metrics():
    data = generate_latest(registry)
    app.logger.debug(f"Metrics, returning: {data}")
    return Response(data, mimetype=CONTENT_TYPE_LATEST)


cron = BackgroundScheduler(daemon=True)
# Explicitly kick off the background thread
cron.start()

TRAIN_INTERVAL_MINUTES = int(os.environ.get("TRAIN_INTERVAL_MINUTES", 30))
# cron.add_job('training_service.train:train', 'interval', minutes=TRAIN_INTERVAL_MINUTES)
cron.add_job(train, "interval", minutes=TRAIN_INTERVAL_MINUTES)
