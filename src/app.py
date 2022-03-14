 #!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify, abort
from ensemble_class import MyEnsembleClassifier
import pandas as pd
import joblib

app = Flask(__name__)

stroke_model = joblib.load("../models/ens.dump")
bmi_model = joblib.load("../models/bmi_model.dump")
glu_model = joblib.load("../models/glu_model.dump")
hyp_model = joblib.load("../models/hyp_model.dump")
glu_bmi_model = joblib.load("../models/glu_bmi_class.dump")
glu_hyp_model = joblib.load("../models/glu_hyp_class.dump")
bmi_hyp_model = joblib.load("../models/bmi_hyp_class.dump")
hyp_bmi_glu_model = joblib.load("../models/hyp_bmi_glu_class.dump")

MODEL_DICT = {
    "stroke": stroke_model,
    "bmi": bmi_model,
    "glu": glu_model,
    "hyp": hyp_model,
    "glu_bmi": glu_bmi_model,
    "glu_hyp": glu_hyp_model,
    "bmi_hyp": bmi_hyp_model,
    "hyp_bmi_glu": hyp_bmi_glu_model,
}
COL_ORDER = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]


def clean_input(input):
    """
    Cleans the data into the same format as the models were trained with.
    Transform JSON into a DataFrame and arranges the columns.
    """
    col_order_list = [col for col in COL_ORDER if col in input.keys()]
    data = pd.DataFrame(input, index=[0]).reindex(columns=col_order_list)
    return data


def get_prediction(model_name, input):
    """
    Return the answer and its probabilities
    for given model and data
    """
    model = MODEL_DICT[model_name]
    data = clean_input(input)
    result = model.predict(data)
    proba = model.predict_proba(data)
    return result, proba


def handle_response(model, input):
    """
    Returns model predictions for given data as JSON
    """
    if model in MODEL_DICT.keys():
        try:
            result, proba = get_prediction(model, input)
            proba = [entry.tolist() for entry in proba]
            return jsonify(prediction=result.tolist(), probability=proba)
        except Exception as e:
            return f"Something went wrong --> {e}"
    else:
        abort(404)


@app.route("/", methods=["POST", "GET"])
def index():
    "Main web app view"
    if request.method == "POST":
        try:
            result, proba = get_prediction("stroke", request.form)
            return render_template(
                "result.html", result=result[0], proba=round(float(proba), 2)
            )
        except Exception as e:
            return f"Something went wrong --> {e}"

    else:
        return render_template("index.html")


@app.route("/from_query/<model>/")
def from_query(model):
    "Handles get request with query"
    return handle_response(model, request.args)


@app.route("/from_json/<model>/", methods=["POST"])
def from_json(model):
    "Handles post request with json"
    return handle_response(model, request.get_json())


if __name__ == "__main__":
    app.run(debug=True)
