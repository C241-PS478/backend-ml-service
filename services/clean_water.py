import numpy as np
import tensorflow as tf

import cv2

def preprocess_image(image, desired_width, desired_height):
    image = cv2.resize(image, (desired_width, desired_height))
    image = image / 255.0
    return image

model = tf.keras.models.load_model("models/clean-water.h5")

def predict(img: cv2.typing.MatLike):

    desired_width = 256
    desired_height = 256

    img = preprocess_image(
        image=img, desired_width=desired_width, desired_height=desired_height
    )

    img = np.expand_dims(img, axis=0) 

    prediction = model.predict(img)

    return prediction