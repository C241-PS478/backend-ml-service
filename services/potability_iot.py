import numpy as np
import tensorflow as tf

import cv2

model = tf.keras.models.load_model("models/potability-iot.h5")

def predict(solids: float, turbidity: float, chloramines: float, organic_carbon: float, sulfate: float, ph: float):
    prediction = model.predict(np.array([[solids, turbidity, chloramines, organic_carbon, sulfate, ph]]))

    return prediction