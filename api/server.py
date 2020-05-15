import joblib
import pandas as pd
import json
import numpy as np
from flask import Flask, jsonify, request


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


app = Flask(__name__)
app.json_encoder = NpEncoder


def get_predict(request, model):
    print(request.values)
    json_ = request.json
    campos = pd.DataFrame(json_)

    if campos.shape[0] == 0:
        return "Dados do usuário estão incorretos.", 400

    for col in model.independentcols:
        if col not in campos.columns:
            campos[col] = 0
    x = campos[model.independentcols]

    prediction = model.predict(x)
    # predict_proba = model.predict_proba(x)
    return prediction


@app.route("/", methods=['GET', 'POST'])
def call_home(request=request):
    print(request.values)
    return "SERVER IS RUNNING"


@app.route("/predict", methods=['POST'])
def call_predict(request=request):
    print(request.values)
    model = request.args.get('model')
    version = request.args.get('version')

    if model == 'gradientboosting_classifier' and version == '1':
        prediction = get_predict(request, model_1_gradientboosting_classifier)
    elif model == 'gradientboosting_classifier' and version == '2':
        prediction = get_predict(request, model_2_gradientboosting_classifier)
    elif model == 'gradientboosting_regressor' and version == '1':
        prediction = get_predict(request, model_1_gradientboosting_regressor)
    elif model == 'gradientboosting_regressor' and version == '2':
        prediction = get_predict(request, model_2_gradientboosting_regressor)

    elif model == 'lightgbm_classifier' and version == '1':
        prediction = get_predict(request, model_1_lightgbm_classifier)
    elif model == 'lightgbm_classifier' and version == '2':
        prediction = get_predict(request, model_2_lightgbm_classifier)
    elif model == 'lightgbm_regressor' and version == '1':
        prediction = get_predict(request, model_1_lightgbm_regressor)
    elif model == 'lightgbm_regressor' and version == '2':
        prediction = get_predict(request, model_2_lightgbm_regressor)

    elif model == 'randomforest_classifier' and version == '1':
        prediction = get_predict(request, model_1_randomforest_classifier)
    elif model == 'randomforest_classifier' and version == '2':
        prediction = get_predict(request, model_2_randomforest_classifier)
    elif model == 'randomforest_regressor' and version == '1':
        prediction = get_predict(request, model_1_randomforest_regressor)
    elif model == 'randomforest_regressor' and version == '2':
        prediction = get_predict(request, model_2_randomforest_regressor)

    elif model == 'xgboost_classifier' and version == '1':
        prediction = get_predict(request, model_1_xgboost_classifier)
    elif model == 'xgboost_classifier' and version == '2':
        prediction = get_predict(request, model_2_xgboost_classifier)
    elif model == 'xgboost_regressor' and version == '1':
        prediction = get_predict(request, model_1_xgboost_regressor)
    elif model == 'xgboost_regressor' and version == '2':
        prediction = get_predict(request, model_2_xgboost_regressor)

    else:
        return "Dados do modelo estão incorretos.", 400

    return {
        'prediction': list(prediction)
    }


if __name__ == '__main__':
    model_1_gradientboosting_classifier = joblib.load(
        'model/model_1_gradientboosting_classifier.joblib')
    model_1_gradientboosting_regressor = joblib.load(
        'model/model_1_gradientboosting_regressor.joblib')
    model_2_gradientboosting_classifier = joblib.load(
        'model/model_2_gradientboosting_classifier.joblib')
    model_2_gradientboosting_regressor = joblib.load(
        'model/model_2_gradientboosting_regressor.joblib')

    model_1_lightgbm_classifier = joblib.load(
        'model/model_1_lightgbm_classifier.joblib')
    model_1_lightgbm_regressor = joblib.load(
        'model/model_1_lightgbm_regressor.joblib')
    model_2_lightgbm_classifier = joblib.load(
        'model/model_2_lightgbm_classifier.joblib')
    model_2_lightgbm_regressor = joblib.load(
        'model/model_2_lightgbm_regressor.joblib')

    model_1_randomforest_regressor = joblib.load(
        'model/model_1_randomforest_regressor.joblib')
    model_1_randomforest_classifier = joblib.load(
        'model/model_1_randomforest_classifier.joblib')
    model_2_randomforest_regressor = joblib.load(
        'model/model_2_randomforest_regressor.joblib')
    model_2_randomforest_classifier = joblib.load(
        'model/model_2_randomforest_classifier.joblib')

    model_1_xgboost_regressor = joblib.load(
        'model/model_1_xgboost_regressor.joblib')
    model_1_xgboost_classifier = joblib.load(
        'model/model_1_xgboost_classifier.joblib')
    model_2_xgboost_regressor = joblib.load(
        'model/model_2_xgboost_regressor.joblib')
    model_2_xgboost_classifier = joblib.load(
        'model/model_2_xgboost_classifier.joblib')

    app.run(port=5000, host='0.0.0.0')
