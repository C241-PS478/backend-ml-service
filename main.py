import pickle
import flask
from flask import request
from services import water_segmentation
import datetime

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return flask.jsonify({
        "message": "WaterWise REST API ML Service",
        "data": {
            "time": datetime.datetime.now()
        }
    })

@app.route('/clean-water', methods=['POST'])
def predict():
    # TODO
    img = None

    prediction = water_segmentation.predict(img)
    
    return flask.jsonify(prediction)

def create_app():
   return app

if __name__ == '__main__':
    app.run(debug=True, port='5000',host='0.0.0.0')