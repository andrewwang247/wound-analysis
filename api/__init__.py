from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
# app.config.from_object('wound_analysis.config')

#app.config.from_object('wound_analysis.config')
#app.config.from_envvar('PROJ_SETTINGS', silent=True)

import api.center
import api.data_process
import api.model
import api.predict
import api.train
import api.routes

