from data_loading import data_loading
import tensorflow as tf # type: ignore
import re

def clean_class_name(class_name: str):
    """
    Convert dataset class name into clean readable format.
    """

    # Split plant and disease
    parts = class_name.split("___")

    if len(parts) == 2:
        plant_raw, disease_raw = parts
    else:
        plant_raw = class_name
        disease_raw = ""

    # Clean plant name
    plant = plant_raw.replace("_", " ")
    plant = plant.replace(",", "")
    plant = re.sub(r"\s+", " ", plant).strip()
    plant = plant.title()

    # Clean disease name
    disease = disease_raw.replace("_", " ")
    disease = re.sub(r"\s+", " ", disease).strip()
    disease = disease.title()

    # Handle special formatting
    disease = disease.replace("Virus", "Virus")
    disease = disease.replace("Mites Two-Spotted Spider Mite",
                              "Spider Mites (Two-Spotted Spider Mite)")

    # Final formatting
    if disease.lower() == "healthy":
        return f"{plant} - Healthy"

    if disease:
        return f"{plant} - {disease}"

    return plant

def data_augmentation():
    
   data_aug = tf.keras.Sequential([
       tf.keras.layers.RandomFlip("horizontal"),   # Randomly flip images
       tf.keras.layers.RandomRotation(0.2),       # Rotate images slightly
       tf.keras.layers.RandomZoom(0.1),           # Apply zoom effect
   ])
   
   _,training_set = data_loading.get_dataset()
   training_set = training_set.map(lambda x,y: (data_aug(x, training=True),y))
   return training_set
