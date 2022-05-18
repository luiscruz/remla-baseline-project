FROM python:3.9-slim

WORKDIR /root/

COPY requirements.txt .

RUN mkdir output &&\
	python -m pip install --upgrade pip &&\
	pip install -r requirements.txt

COPY src src
COPY data data

# TODO needs the structure already in place to work
RUN python src/read_data.py &&\
	python src/text_preprocessing.py &&\
	python src/text_classification.py

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["src/serve_model.py"]