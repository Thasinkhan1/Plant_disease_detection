#!/usr/bin/env python
# coding: utf-8

# In[1]:


import kaggle as kg
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from keras.applications import VGG16
from glob import glob
from keras.utils import to_categorical
from keras.layers import  Dense,Conv2D,BatchNormalization,GlobalAveragePooling2D,MaxPooling2D,Flatten,Dropout
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.initializers import he_normal
from keras.regularizers import l1, l2
plt.style.use("ggplot")


# In[2]:


def count_folders(directory):
    folder_count = 0
    for entry in os.scandir(directory):
        if entry.is_dir():
            folder_count += 1
    return folder_count

def count_images(directory):
    images_count = 0
    
    for root,dir, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(("jpeg","png","jpg")):
                images_count = images_count + 1 
    return images_count
# Function to count the total number of files in a directory
def total_files(folder_path):
    num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    return num_files


train = "/home/thasin/plant_disease/Dataset/train"
valid = "/home/thasin/plant_disease/Dataset/valid"
test = "/home/thasin/plant_disease/Dataset/test"

# Count the number of folders in train and valid directories
total_folders_train = count_folders(train)
total_folders_valid = count_folders(valid)

# Count the total number of files in the test directory
total_files_test = total_files(test)

print(f"Total number of folders in Train: {total_folders_train} and images in the folder is {count_images(train)}")
print(f"Total number of folders in Valid: {total_folders_valid} and images in that folder is {count_images(valid)}")
print(f"Total number of files in Test: {total_files_test} and images in that folder is {count_images(test)}")


# In[3]:


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


# In[4]:


train_path = "/home/thasin/plant_disease/Dataset/train"
valid_path = "/home/thasin/plant_disease/Dataset/valid"
test_path = "/home/thasin/plant_disease/Dataset/test"


# In[5]:


train_path


# In[6]:


train_df = train_test_df(train_path,is_test=False)
valid_df = train_test_df(valid_path,is_test=False)
test_df = train_test_df(test_path,is_test=True)


# In[7]:


train_df.shape


# In[8]:


train_df.head(10)


# In[9]:


valid_df.shape


# In[10]:


valid_df.head(10)


# In[11]:


convert_to_int = dict(zip(train_df["label"].unique(),range(len(train_df["label"].unique()))))


# In[12]:


convert_to_int


# In[13]:


range(len(train_df['label'].unique()))


# In[14]:


train_df["label"].replace(to_replace=convert_to_int.keys(),value=convert_to_int.values(),inplace=True)


# In[15]:


train_df.head(700)


# In[16]:


valid_df.replace(to_replace=convert_to_int.keys(),value=convert_to_int.values(),
                     inplace=True)


# In[17]:


valid_df.tail(2000)


# In[18]:


range(len(train_df["label"].unique())), range(len(valid_df["label"].unique()))


# In[19]:


train_df["label"].unique


# In[20]:


# valid_df["label"].unique


# In[21]:


# valid_df = valid_df[valid_df["label"].isin(range(5))]


# In[22]:


Y_true_train = to_categorical(y=train_df["label"],num_classes=6)
Y_true_test = to_categorical(y=valid_df["label"],num_classes=6)


# In[23]:


Y_true_train


# In[24]:


Y_true_train.shape, Y_true_test.shape


# In[25]:


def plant_disease_cnn():

    vgg16 = VGG16(include_top=False,input_shape=(224,224,3),weights="imagenet",pooling=None)
    vgg16.trainable = False
    input_to_vgg16 = vgg16.input
    vgg16_output = vgg16.output
    
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01))(vgg16_output)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = GlobalAveragePooling2D()(x)  # formula = output_c = 1 / (H * W) * sum_{i=1}^{H} sum_{j=1}^{W} X[i, j, c]
    
    x = Dense(128,activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(64,activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    
    vgg16_output = Dense(6,activation='softmax', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01))(x)
    
    model2 = Model(inputs=[input_to_vgg16], outputs=[vgg16_output])
    
    return model2


# In[26]:


model = plant_disease_cnn()
model.summary()


# In[27]:


model.get_weights()


# In[28]:


# # Compile the Model
# model.compile(optimizer='SGD',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


# In[29]:


def custom_data_generator(data_df,Y_true,mb_size):
    
    for time_step in range(data_df.shape[0]//mb_size):
        X_mb = list()
        for img_path in data_df.iloc[time_step*mb_size:(time_step+1)*mb_size,0]:
            img_np_array = plt.imread(img_path)
            reshaped_np_array = cv2.resize(img_np_array, (224, 224))
            X_mb.append(reshaped_np_array)
        X_mb = np.array(X_mb)
        Y_true_mb = Y_true[time_step*mb_size:(time_step+1)*mb_size]
        
        yield X_mb, Y_true_mb


# In[30]:


# # Assume train_df, Y_true_train, valid_df, Y_true_test are defined properly
# batch_size = 64
# epochs = 10

# model.fit(
#     custom_data_generator(train_df, Y_true_train, batch_size),
#     validation_data=custom_data_generator(valid_df, Y_true_test, batch_size),
#     epochs=epochs,
#     steps_per_epoch=len(train_df) // batch_size,
#     # validation_steps=len(valid_df) // batch_size
# )


# In[31]:


epochs = 20
training_data_mb_size = 64
testing_data_mb_size = 64


# In[32]:


def loss_fn(Y_true_mb,Y_pred_mb):
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_true_mb,y_pred=Y_pred_mb))
optimizer = Adam(learning_rate=0.001)


# In[33]:


@tf.function
def training_step(X_train_mb,Y_true_train_mb):

    with tf.GradientTape() as tape:
            
        Y_pred_train_mb = model(X_train_mb, training=True)
        training_loss = loss_fn(Y_true_train_mb, Y_pred_train_mb)

    grads = tape.gradient(training_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(Y_true_train_mb,Y_pred_train_mb)

    return training_loss


# In[34]:


@tf.function
def testing_forward_pass(X_test_mb,Y_true_test_mb):

    Y_pred_test_mb = model(X_test_mb,training=False)
    testing_loss = loss_fn(Y_true_test_mb,Y_pred_test_mb)
    test_acc_metric.update_state(Y_true_test_mb,Y_pred_test_mb)

    return testing_loss


# In[35]:


train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

for epoch in range(epochs):

    training_data_generator = custom_data_generator(train_df,Y_true_train,64)

    for time_step, (X_train_mb, Y_true_train_mb) in enumerate(training_data_generator):
        training_loss = training_step(X_train_mb,Y_true_train_mb)

        if (time_step+1) % 10 == 0:
            print("Epoch %d, Time Step %d, Training loss for one mini batch: %.4f"
            % (epoch+1, time_step+1, float(training_loss)))
            
    training_acc = train_acc_metric.result()    
    print("Epoch %d, Training Accuracy: %.2f" % (epoch+1,float(training_acc)))
    train_acc_metric.reset_states()

    testing_data_generator = custom_data_generator(valid_df,Y_true_test,testing_data_mb_size)

    for X_test_mb, Y_true_test_mb in testing_data_generator:
        testing_loss = testing_forward_pass(X_test_mb,Y_true_test_mb)

    print("\nEpoch %d, Testing Loss for last mini batch: %.4f" % (epoch+1,float(testing_loss)))
    testing_acc = test_acc_metric.result()
    print("Epoch %d, Testing Accuracy: %.2f" % (epoch+1,float(testing_acc)))
    test_acc_metric.reset_states()

    print("\n\n")


# In[36]:


model.save_weights('model_weights.h5')  # Save weights in HDF5 format


# In[37]:


plt.figure(figsize=(12,6))
plt.subplot(1,2,2)
plt.plot(epochs, training_loss, label="training_loss",marker='o')
plt.plot(epochs, testing_loss, label="testing_loss",marker='o')
plt.title("Loss Comparosion")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[36]:


model.load_weights("model_weights.h5")


# In[53]:


from keras.preprocessing.image import load_img, img_to_array

# Load and preprocess the image
image_path = '/home/thasin/plant_disease/Dataset/train/Apple___Black_rot/0b8dabb7-5f1b-4fdc-b3fa-30b289707b90___JR_FrgE.S 3047_270deg.JPG'
img = load_img(image_path, target_size=(224,224))  # Resize to model input size
img_array = img_to_array(img) / 255.0  # Normalize pixel values
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension


# In[ ]:


# Make predictions
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)  # Get the class with the highest probability
print(f"Predicted Class: {predicted_class[0]}")


# In[41]:


# Map the predicted class to label
class_labels = ['Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Apple___Black_rot', 'Apple___Apple_scab', 'pple___Cedar_apple_rust']  # Example labels
print(f"Predicted Label: {class_labels[predicted_class[0]]}")


# In[42]:


plt.imshow(img)
plt.title(f"Predicted: {class_labels[predicted_class[0]]}")
plt.axis('off')
plt.show()


# In[ ]:


# Sample knowledge base for disease prevention advice
disease_advice = {
    "Powdery Mildew": "To prevent powdery mildew, ensure good air circulation around plants, avoid overhead watering, and use resistant plant varieties.",
    "Leaf Spot": "Prevent leaf spot by avoiding overhead irrigation, removing affected leaves, and applying fungicides if necessary.",
    "Root Rot": "To prevent root rot, ensure proper drainage in pots and soil, avoid overwatering, and regularly check for root health.",
    "Blight": "Prevent blight by practicing crop rotation, using resistant varieties, and avoiding wet foliage.",
    # Add more diseases and their prevention advice
}

def get_disease_advice(disease_name):
    # Fetch advice based on the detected disease
    return disease_advice.get(disease_name, "No specific advice available for this disease.")

# Example usage
detected_disease = "Powdery Mildew"  # This would be the output of your model
advice = get_disease_advice(detected_disease)
print(f"Detected Disease: {detected_disease}")
print(f"Prevention Advice: {advice}")


# In[ ]:


# # Assume 'model' is your trained model and 'predict' is your function for prediction
# def predict_and_advise(image):
#     detected_disease = model.predict(image)  # Get the disease from the model
#     advice = get_disease_advice(detected_disease)
#     return detected_disease, advice

# # Example usage
# image = "path/to/plant/image.jpg"
# disease, advice = predict_and_advise(image)
# print(f"Detected Disease: {disease}")
# print(f"Prevention Advice: {advice}")


# In[162]:


# Epoch 1, Time Step 10, Training loss for one mini batch: 2.7211
# Epoch 1, Training Accuracy: 0.34

# Epoch 1, Testing Loss for last mini batch: 20.4809
# Epoch 1, Testing Accuracy: 0.20



# Epoch 2, Time Step 10, Training loss for one mini batch: 1.3338
# Epoch 2, Training Accuracy: 0.43

# Epoch 2, Testing Loss for last mini batch: 4.9933
# Epoch 2, Testing Accuracy: 0.20



# Epoch 3, Time Step 10, Training loss for one mini batch: 1.2735
# Epoch 3, Training Accuracy: 0.41

# Epoch 3, Testing Loss for last mini batch: 3.6778
# Epoch 3, Testing Accuracy: 0.20



# Epoch 4, Time Step 10, Training loss for one mini batch: 0.9547
# Epoch 4, Training Accuracy: 0.41

# Epoch 4, Testing Loss for last mini batch: 3.6600
# Epoch 4, Testing Accuracy: 0.20



# Epoch 5, Time Step 10, Training loss for one mini batch: 1.0383
# Epoch 5, Training Accuracy: 0.41

# Epoch 5, Testing Loss for last mini batch: 3.0552
# Epoch 5, Testing Accuracy: 0.20



# Epoch 6, Time Step 10, Training loss for one mini batch: 0.9107
# Epoch 6, Training Accuracy: 0.43

# Epoch 6, Testing Loss for last mini batch: 2.5043
# Epoch 6, Testing Accuracy: 0.20



# Epoch 7, Time Step 10, Training loss for one mini batch: 0.8479
# Epoch 7, Training Accuracy: 0.51

# Epoch 7, Testing Loss for last mini batch: 2.7955
# Epoch 7, Testing Accuracy: 0.22



# Epoch 8, Time Step 10, Training loss for one mini batch: 0.8440
# Epoch 8, Training Accuracy: 0.55

# Epoch 8, Testing Loss for last mini batch: 2.8175
# Epoch 8, Testing Accuracy: 0.29



# Epoch 9, Time Step 10, Training loss for one mini batch: 0.6454
# Epoch 9, Training Accuracy: 0.64

# Epoch 9, Testing Loss for last mini batch: 2.7031
# Epoch 9, Testing Accuracy: 0.36



# Epoch 10, Time Step 10, Training loss for one mini batch: 0.5065
# Epoch 10, Training Accuracy: 0.70

# Epoch 10, Testing Loss for last mini batch: 2.5889
# Epoch 10, Testing Accuracy: 0.46



# Epoch 11, Time Step 10, Training loss for one mini batch: 0.5639
# Epoch 11, Training Accuracy: 0.76

# Epoch 11, Testing Loss for last mini batch: 2.4106
# Epoch 11, Testing Accuracy: 0.56



# Epoch 12, Time Step 10, Training loss for one mini batch: 0.3706
# Epoch 12, Training Accuracy: 0.79

# Epoch 12, Testing Loss for last mini batch: 2.3900
# Epoch 12, Testing Accuracy: 0.57



# Epoch 13, Time Step 10, Training loss for one mini batch: 0.3051
# Epoch 13, Training Accuracy: 0.86

# Epoch 13, Testing Loss for last mini batch: 2.4279
# Epoch 13, Testing Accuracy: 0.69



# Epoch 14, Time Step 10, Training loss for one mini batch: 0.4168
# Epoch 14, Training Accuracy: 0.87

# Epoch 14, Testing Loss for last mini batch: 2.6288
# Epoch 14, Testing Accuracy: 0.70



# Epoch 15, Time Step 10, Training loss for one mini batch: 0.1479
# Epoch 15, Training Accuracy: 0.90

# Epoch 15, Testing Loss for last mini batch: 3.1394
# Epoch 15, Testing Accuracy: 0.71

#i got by using learning rate of 0.01 with adam and using he_uniform and l2 reguralization


# In[ ]:





# In[ ]:




