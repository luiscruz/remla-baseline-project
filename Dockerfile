FROM python:3.9-slim

WORKDIR /root/

COPY pyproject.toml .

RUN mkdir output &&\
	python -m pip install --upgrade pip &&\
	pip install poetry &&\
	poetry install --no-dev

COPY src src
COPY data data

RUN poetry run python src/preprocess.py &&\
	poetry run python src/train.py

EXPOSE 8080

ENTRYPOINT ["poetry", "run"]
CMD ["python", "src/serve.py"]
