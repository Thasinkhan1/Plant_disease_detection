from flask import Flask, render_template, request, jsonify, url_for
# from keras.models import load_model
# from src.main import plant_disease_cnn
from keras.preprocessing.image import img_to_array
import numpy as np

app = Flask(__name__)

# model = plant_disease_cnn()
# model.load_weights('model_weights.h5')

@app.route("/", methods=["GET", "POST"])
def welcome():
    if request.method == "POST":
        if 'plant_image' in request.files: 
            file = request.files("Plant_Image")
            
            from PIL import Image
            image = Image.open(file)
            image = image.resize(128,128)
            image = img_to_array(image) / 255.0 #normalize pixel value
            image = np.expand_dims(image, axis=0) #add batch diamensions
           
            # #for prediction 
            # prediction = model.predict(image)
            # prediction_class = np.argmax(prediction)
            # accuracy = np.max(prediction) * 100
            
            disease_classes = {
                0: "Blueberry healthy",
                1: "Cherry Healthy",
                2: "Apple black rot",
                3: "Apple Scab",
                4: "Apple healthy",
                5: "Apple cedar rust"
                
            }
            
            advice_map = {
                0: "None",
                1: "None",
                2: "None",
                3: "None",
                4: "None",
                5: "None",
            }
            
            disease = disease_classes.get(prediction_class,"Unkown_diease")
            advice = advice_map.get(prediction_class,"No advice available")
            
            return render_template('form.html', accuracy=round(accuracy,2), disease=disease, advice= advice)
    
        return "No file uploaded", 400

    # Default GET request - render the form
    return render_template('form.html', accuracy=None, disease=None, advice=None)




@app.route("/success", methods=["POST"])
def success():
    
    if 'plant_image' not in request.files:
        return 'No file uploaded', 400
    
    accuracy = 95.0 #assuming the accuracy
    
    return render_template('form.html',accuracy=accuracy)
# API Route for JSON
@app.route('/api', methods=['POST'])
def advice():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Example advice logic
    advice_text = "Make sure to water the plant regularly."
    return jsonify({"advice": advice_text})

if __name__ == "__main__":
    app.run(debug=True)
