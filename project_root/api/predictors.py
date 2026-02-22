import tensorflow as tf
import numpy as np
import os
import gdown
from pathlib import Path
from tensorflow.keras.models import load_model

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.h5"

MODEL = None

def download_model():
    """Download model from Google Drive if not exists"""
    if not MODEL_PATH.exists():
        print("Downloading model from Google Drive...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        file_id = "1c0QDmCBUdBnXHUCslkf1J0ChkI-rhewZ"
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, str(MODEL_PATH), quiet=False)

def load_model_once():
    """Load model into memory"""
    global MODEL
    if MODEL is None:
        download_model()
        print("Loading model...")
        MODEL = load_model(MODEL_PATH)
        print("Model loaded successfully.")

def predict(image):
    """Run prediction"""
    global MODEL

    if MODEL is None:
        load_model_once()

    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)

    predictions = MODEL.predict(input_arr)
    return predictions