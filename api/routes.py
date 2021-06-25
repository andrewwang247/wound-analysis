# [START gae_python38_app]

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
#from flask import current_app
import api
import flask
from api.predict import predict
from PIL import Image
import io
import base64
import pathlib

@api.app.route('/predictMask', methods=['POST', 'GET'])
def predict_mask():
    input_json = flask.request.get_json(force=True)
    b64_string = input_json["b64img"]
    decoded = base64.b64decode(b64_string)
    request_image = io.BytesIO(decoded)
    opened_image = Image.open(request_image)
    model_path = pathlib.Path(__file__).resolve().parent

    prediction = predict(opened_image, model_path)
    print("prediction:", prediction)
    return str(prediction)

@api.app.route("/", methods=["POST", "GET"])
def base():
    return "This is the ml api"