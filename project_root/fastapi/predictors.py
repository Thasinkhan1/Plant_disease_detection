import os
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from project_root.config import config
import requests

MODEL_PATH = "saved_model/model.h5"
MODEL = None

# def download_model():
#     if not os.path.exists(MODEL_PATH):
#         url = "https://your-bucket.com/model.h5"  #drive path
#         os.makedirs("saved_model", exist_ok=True)
#         with open(MODEL_PATH, 'wb') as f:
#             f.write(requests.get(url).content)

# def load_model():
#     global MODEL
MODEL = tf.keras.models.load_model(config.SAVED_MODEL_PATH)


def predict(image):
    image = image.resize((128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = MODEL.predict(input_arr)
    return predictions
