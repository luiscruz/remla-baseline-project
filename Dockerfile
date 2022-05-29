FROM python:3.9.12-slim

WORKDIR /root/

COPY requirements.txt .
COPY setup.py .
COPY src src
COPY models models
COPY data data

RUN python -m pip install --upgrade pip &&\
    pip install -r requirements.txt &&\
    pip install -e .

# Seperate RUN commands to enable caching the different stages
RUN python src/data/make_dataset.py
RUN python src/features/build_features.py
RUN python src/models/train_model.py

# Dynamic interaction with the model
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["src/models/serve_model.py"]
