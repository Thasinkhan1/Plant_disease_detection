from data_loading import data_loading
import tensorflow as tf # type: ignore

def data_augmentation():
    
   data_aug = tf.keras.Sequential([
       tf.keras.layers.RandomFlip("horizontal"),   # Randomly flip images
       tf.keras.layers.RandomRotation(0.2),       # Rotate images slightly
       tf.keras.layers.RandomZoom(0.1),           # Apply zoom effect
   ])
   
   _,training_set = data_loading.get_dataset()
   training_set = training_set.map(lambda x,y: (data_aug(x, training=True),y))
   return training_set
