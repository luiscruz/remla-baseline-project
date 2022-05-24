FROM python:3.10.4-slim

WORKDIR /root/

COPY requirements.txt .

RUN mkdir output &&\
	python -m pip install --upgrade pip &&\
    sed -i '/mllint/d' ./requirements.txt &&\
	pip install -r requirements.txt

COPY src src
COPY data data

RUN python src/preprocess.py &&\
	python src/train.py

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["src/serve.py"]
