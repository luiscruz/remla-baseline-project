import os
import time
from threading import Thread
from typing import Tuple

import pandas as pd
import requests
from flask import Flask, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Summary,
    generate_latest,
)

import src.scraper.app.data_validation as data_validation

app_name = "scraping-service"
app = Flask(app_name)
app.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

registry = CollectorRegistry()

num_queries = Summary("get_query", "Get query function")
scrape_metric = Summary("scrape", "Scrape stackoverflow function")
question_count = Counter("num_questions_retrieved", "Number of questions scraped from SO")


@app.route("/metrics")
def metrics():
    data = generate_latest(registry)
    app.logger.debug(f"Metrics, returning: {data}")
    return Response(data, mimetype=CONTENT_TYPE_LATEST)


def get_query(dateFrom, dateTo, page=1):
    return (
        f"https://api.stackexchange.com/2.3/questions?"
        f"page={page}&"
        f"pagesize=100&"
        f"fromdate={dateFrom}&"
        f"todate={dateTo}&"
        f"order=desc&"
        f"sort=activity&"
        f"site=stackoverflow&"
        f"filter=!Fc7.FlqcJXCgmWba*Q45*UiJ(2"
    )


@num_queries.time()
def execute_query(query) -> Tuple[bool, dict]:
    res = requests.get(query)
    success = False
    if res:
        backoff = res.json().get("backoff", 0)
        if backoff > 0:
            app.logger.debug(f"Waiting for {backoff} seconds before continuing")
            time.sleep(backoff)
        success = True
    else:
        app.logger.debug(f"query got non OK response: \n{res.status_code = }, {res.json() = },\nsleeping for 60 secs")
        time.sleep(60)
    return success, res.json()


@scrape_metric.time()
def scrape_questions_and_save(fromdate: str, todate: str, apikey=None, save_dir=""):
    # TODO use api key
    app.logger.debug("Scrape and save")
    # Request data
    page = 1
    success, response_dict = execute_query(get_query(fromdate, todate, page=page))
    if not success:
        df = pd.DataFrame()
    else:
        items = response_dict["items"]
        while response_dict["has_more"]:
            items.append(response_dict["items"])
            success, response_dict = execute_query(get_query(fromdate, todate, page=page + 1))

        df = pd.DataFrame(items)

    # transform to dataframe and store as tsv file
    if not df.empty:
        df = df[["title", "tags"]]
        app.logger.debug(f"Removing anomalies from df with shape: {df.shape}")
        try:
            num_anomalies, df = data_validation.remove_anomalies(df)
        except Exception as e:
            app.logger.warning(f"Exception thrown during data validation, ignoring data: \n{e}")
            num_anomalies = 1

        if num_anomalies == 0:
            question_count.inc(len(df))
            file_name = f"{save_dir}/result_{fromdate}-{todate}.tsv"
            app.logger.debug(f"Saving to {file_name}")
            df.to_csv(file_name, sep="\t", index=False)
            app.logger.debug(f"{len(df)} questions scraped")
        else:
            app.logger.warning("Anomalies found, not saving results")
    else:
        app.logger.info("Dataframe result empty (no questions found)")


controller_host = os.environ["CONTROLLER_HOST"]
save_dir = os.environ["SCRAPE_SAVE_DIR"]


def scrape_loop():
    app.logger.info("Scrape loop started")
    apikey = None
    quota_remaining = None
    while True:
        app.logger.debug("Querying")
        # Request params from Scraper controller
        params = ""
        if apikey and quota_remaining:
            params = f"/{apikey}/{quota_remaining}"
        url = f"{controller_host}/date_range{params}"
        response = requests.get(url)
        if response:
            scrape_questions_and_save(**response.json(), save_dir=save_dir)
        else:
            app.logger.debug(
                f"Response code for URL: {url}\n"
                f"was error code: {response}, {response.text} "
                f"sleeping for 1 minute before retrying"
            )
            time.sleep(60)


main_loop = Thread(target=scrape_loop)
main_loop.start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)  # nosec
