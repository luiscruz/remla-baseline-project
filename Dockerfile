FROM python:3.7.10

RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& apt-get -y install gcc mono-mcs \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /root/

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip \
&& pip3 install -r requirements.txt

CMD "bash"
