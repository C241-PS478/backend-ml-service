import flask
import datetime
import numpy as np
import cv2
from io import BytesIO

from services import clean_water, water_segmentation, potability_iot


app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return flask.jsonify({
        "message": "WaterWise ML Service",
        "data": {
            "time": datetime.datetime.now()
        }
    })

@app.route('/water-segmentation', methods=['POST'])
def get_extracted_water():

    img = flask.request.files.get('image', None)
    if not img:
        response = {
            "message": "No image found."
        }
        return flask.jsonify(response), 400
    
    file_stream = BytesIO(img.read())
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_extracted, img_overlay = water_segmentation.extract(img)

    if img_extracted is None:
        response = {
            "message": "Not enough water is detected."
        }
        return flask.jsonify(response), 400

    retval_extracted, buffer_extracted = cv2.imencode('.png', img_extracted)
    
    return flask.send_file(
        BytesIO(buffer_extracted),
        mimetype='image/png'
    )

@app.route('/water-segmentation/overlay', methods=['POST'])
def get_water_overlay():

    img = flask.request.files.get('image', None)
    if not img:
        response = {
            "message": "No image found."
        }
        return flask.jsonify(response), 400
    
    file_stream = BytesIO(img.read())
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_extracted, img_overlay = water_segmentation.extract(img)

    retval_overlay, buffer_overlay = cv2.imencode('.png', img_overlay)
    
    return flask.send_file(
        BytesIO(buffer_overlay),
        mimetype='image/png'
    )

@app.route('/clean-water', methods=['POST'])
def predict_clean_water():

    img = flask.request.files.get('image', None)
    if not img:
        response = {
            "message": "No image found."
        }
        return flask.jsonify(response), 400
    
    file_stream = BytesIO(img.read())
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    prediction = clean_water.predict(img)

    return flask.jsonify({
        "message": "Prediction successful.",
        "data": {
            "prediction": float(prediction[0][0])
        }
    })

@app.route('/clean-water/with-extraction', methods=['POST'])
def predict_clean_water_full():

    img = flask.request.files.get('image', None)
    if not img:
        response = {
            "message": "No image found."
        }
        return flask.jsonify(response), 400
    
    file_stream = BytesIO(img.read())
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_extracted, overlay_img = water_segmentation.extract(img)

    if img_extracted is None:
        response = {
            "message": "Not enough water is detected."
        }
        return flask.jsonify(response), 400

    prediction = clean_water.predict(img_extracted)

    return flask.jsonify({
        "message": "Prediction successful.",
        "data": {
            "prediction": float(prediction[0][0])
        }
    })

@app.route('/potability-iot', methods=['POST'])
def predict_potability_iot():

    if flask.request.form:
        data = flask.request.form
    else:
        data = flask.request.json

    print(data)
    print(data['solids'])
    print(data['turbidity'])
    print(data['chloramines'])
    print(data['organic_carbon'])
    print(data['sulfate'])
    print(data['ph'])
    

    prediction = potability_iot.predict(
        solids=float(data['solids']),
        turbidity=float(data['turbidity']),
        chloramines=float(data['chloramines']),
        organic_carbon=float(data['organic_carbon']),
        sulfate=float(data['sulfate']),
        ph=float(data['ph'])
    )

    return flask.jsonify({
        "message": "Prediction successful.",
        "data": {
            "prediction": float(prediction[0][0])
        }
    })

def create_app():
   return app

if __name__ == '__main__':
    app.run(debug=True, port='5000',host='0.0.0.0')