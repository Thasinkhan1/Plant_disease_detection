import pandas as pd
import pathlib
from config import config
import tensorflow as tf

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

TRAIN_DF  = train_test_df(config.TRAIN_PATH, is_test=False)
VALID_DF  = train_test_df(config.VALID_PATH, is_test=False)
TEST_DF  = train_test_df(config.TEST_PATH, is_test=True)

training_set = tf.keras.utils.image_dataset_from_directory(
    config.TRAIN_PATH,
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
    config.VALID_PATH,
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