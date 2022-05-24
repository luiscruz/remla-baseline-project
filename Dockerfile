FROM python:3.9.10-slim

RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& apt-get -y install gcc mono-mcs \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /root/


COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

EXPOSE 8080

CMD "bash"
