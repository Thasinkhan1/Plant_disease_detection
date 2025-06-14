from config import config
from keras.models import load_model # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore

def predict(test_image):
    model = load_model(config.SAVED_MODEL_PATH) 
    image_path = test_image
    image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    model_prediction = config.CLASS_NAMES[result_index]
    
    return f"Disease Name: {model_prediction}:  With confidence: {confidence:.2f}%"
