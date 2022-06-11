FROM python:3.8.13-buster

WORKDIR /root/

COPY src/scraping_service/app src/scraping_service/app
COPY src/__init__.py src/gunicorn_config.py src/startup.sh src/

RUN python -m pip install --upgrade pip && \
    python -m pip install -r src/scraping_service/app/requirements.txt

expose 5000