import os
import shutil
import time

from flask import Flask, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Summary,
    generate_latest,
    multiprocess,
)

PROMETHEUS_MULTIPROC_DIR = os.environ["PROMETHEUS_MULTIPROC_DIR"]
# make sure the dir is clean
shutil.rmtree(PROMETHEUS_MULTIPROC_DIR, ignore_errors=True)
os.makedirs(PROMETHEUS_MULTIPROC_DIR)

app_name = "scraping-controller-service"
app = Flask(app_name)
app.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

duration_metric = Summary("get_timerange_duration", "Time spent on training")
timestamp_metric = Gauge("scrape_timestamp", "Latest epoch seconds given out by scraper controller")

api_keys = {key: 10_000 for key in os.environ["API_KEYS"].split(",")}
api_key_counters = {key: Gauge(f"api_key_{i}_quota", "Remaining quota per api key") for i, key in enumerate(api_keys)}
CURRENT_TIMESTAMP = int(os.environ.get("SCRAPE_START_TIMESTAMP", time.time() - 3600 * 24))
SCRAPE_INCREMENT = int(os.environ.get("SCRAPE_INCREMENT_SECONDS", 300))


def get_next_apikey():
    return max(api_keys, key=api_keys.get)


@app.route("/date_range", methods=["GET"])
@app.route("/date_range/<apikey>/<quota_remaining>", methods=["GET"])
@duration_metric.time()
def get_new_date_range(apikey=None, quota_remaining=None):
    global CURRENT_TIMESTAMP
    if apikey and quota_remaining:
        api_keys[apikey] = quota_remaining
        api_key_counters[apikey].set(quota_remaining)
    res, status_code = "No timerange available", 400
    if time.time() - SCRAPE_INCREMENT > CURRENT_TIMESTAMP:
        res = {
            "fromdate": CURRENT_TIMESTAMP,
            "todate": CURRENT_TIMESTAMP + SCRAPE_INCREMENT,
            "apikey": get_next_apikey(),
        }
        status_code = 200

    CURRENT_TIMESTAMP += SCRAPE_INCREMENT
    timestamp_metric.set(CURRENT_TIMESTAMP)
    app.logger.debug(f"Returning new date range: {res}")
    return res, status_code


@app.route("/metrics")
def metrics():
    data = generate_latest(registry)
    app.logger.debug(f"Metrics, returning: {data}")
    return Response(data, mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(port=5000)
