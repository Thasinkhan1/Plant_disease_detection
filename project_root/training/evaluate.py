from config import config
from data_loading import data_loading
from data_transformation import data_augmentation
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore

def evaluate_model():
    model = load_model("/home/thasin/plant_disease/project_root/models/models.h5")    
    
    #Training set Accuracy
    train_loss, train_acc = model.evaluate(data_augmentation.data_augmentation)
    print(f"\nTraining Accuracy: {train_acc:.4f}, Training Loss: {train_loss:.4f}")

    #Validation set Accuracy
    valid_loss, valid_acc = model.evaluate(data_loading.validation_set)

    print(f"Validation Accuracy: {valid_acc:.4f}, Validation Loss: {valid_loss:.4f}")

# if __name__ == "__main__":
    evaluate_model()