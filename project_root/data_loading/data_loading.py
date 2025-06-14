import pandas as pd # type: ignore
import pathlib
from config import config
import tensorflow as tf # type: ignore
import os
import gc
def train_test_df(path,is_test=False):

    img_path = list()
    img_label = list()

    
    if is_test:
       
        for img_file_path in pathlib.Path(path).glob("*.JPG"):
            img_path.append(str(img_file_path))  # Store the image path
            
            # Assuming label is derived from the filename, for example, before the first underscore
            img_label.append(str(img_file_path.stem).split("_")[0])
            
    else:        
       
       for single_class_dir_path in pathlib.Path(path).glob("*"):
           
           if single_class_dir_path.is_dir():
               
               label = single_class_dir_path.stem
               
               for img_file_path in pathlib.Path(single_class_dir_path).glob("*.[jJ][pP][gG]"):
       
                    img_path.append(str(img_file_path))  # Store the image path
                    # Assuming label is derived from the filename, for example, before the first underscore
                    img_label.append(label)

    return pd.DataFrame(data={"img_path":img_path,"label":img_label})    

#train_path =  os.path.abspath(config.TRAIN_PATH)
def get_dataset():
     train_path =  "Dataset/train"
     valid_path = "Dataset/valid"
     test_path = "Dataset/valid"
     
     TRAIN_DF  = train_test_df(train_path, is_test=False)
     VALID_DF  = train_test_df(valid_path, is_test=False)
     TEST_DF  = train_test_df(test_path, is_test=True)
     
     training_set = tf.keras.utils.image_dataset_from_directory(
         train_path,
         labels="inferred",
         label_mode="categorical",
         class_names=None,
         color_mode=config.COLOR_RGB,
         batch_size=config.BATCH_SIZE,
         image_size=config.IMAGE_SIZE,
         shuffle=True,
         seed=None,
         validation_split=None,
         subset=None,
         interpolation=config.INTERPOLATION,
         follow_links=False,
         crop_to_aspect_ratio=False
     )
     
     
     validation_set = tf.keras.utils.image_dataset_from_directory(
         valid_path,
         labels="inferred",
         label_mode="categorical",
         class_names=None,
         color_mode=config.COLOR_RGB,
         batch_size=config.BATCH_SIZE,
         image_size=config.IMAGE_SIZE,
         shuffle=True,
         seed=None,
         validation_split=None,
         subset=None,
         interpolation=config.INTERPOLATION,
         follow_links=False,
         crop_to_aspect_ratio=False
     )
     
     
     return validation_set, training_set
#if __name__ == "__main__":        
#         print(TRAIN_DF.head())
#     for images, labels in training_set.take(1):
#         print("Batch shape:", images.shape)
#         print("Labels shape:", labels.shape)
#         print("First batch of labels:", labels.numpy())
#     del training_set
#     del validation_set
#     gc.collect()