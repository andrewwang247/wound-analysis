# [START gae_python38_app]

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
#from flask import current_app
import api
import flask
from api.predict import predict
from PIL import Image

@api.app.route('/predictMask', methods=['POST', 'GET'])
def predict_mask():
    request_image = flask.request.args.get("b64img")
    print("req img:", request_image)
    opened_image = Image.open(request_image)
    print("opened img:", opened_image)
    prediction = predict(opened_image)
    print("prediction:", prediction)
    return prediction

@api.app.route("/", methods=["POST", "GET"])
def base():
    return "This is the ml api"