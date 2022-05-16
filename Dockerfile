FROM python:3.7.13-slim

WORKDIR /root/

COPY requirements.txt .

RUN mkdir output &&\
	python -m pip install --upgrade pip &&\
	pip install -r requirements.txt

COPY src src
COPY data data

RUN python src/1_preprocessing.py &&\
	python src/2_bagOfWords.py &&\
	python src/3_train.py

ENTRYPOINT ["python"]
CMD ["4_predict.py"]