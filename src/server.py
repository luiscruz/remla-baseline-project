from flask import Flask, request
from persistence import load_models
from infer import infer

app = Flask('inference-api')

models = load_models('model.joblib')


@app.get('/')
@app.get('/ping')
def ping():
    return 'pong'


@app.post('/infer')
def inference():
    data = request.json
    questions = data['questions']
    preds, _ = infer(questions, *models)
    preds = [list(i) for i in preds]
    return {'tags': preds}
