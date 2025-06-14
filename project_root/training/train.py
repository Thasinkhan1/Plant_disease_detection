from config import config
import tensorflow as tf # type: ignore
from keras.models import load_model # type: ignore
from keras.models import Model # type: ignore
from keras.layers import  Dense,Conv2D,BatchNormalization,Dropout # type: ignore
from data_transformation import data_augmentation
from data_loading import data_loading


def plant_disease_cnn():

    vgg16 = config.MODEL(include_top=False,input_shape=config.INPUT_SHAPE,weights="imagenet",pooling=None)
    vgg16.trainable = True
    
    for layer in vgg16.layers[:-4]:
        layer.trainable = False # Freeze all but the last 4 layers
        
    input_to_vgg16 = vgg16.input
    vgg16_output = vgg16.output
    
    x = Conv2D(filters=64, kernel_size=config.KERNEL_SIZE, activation=config.HIDDEN_ACTIVATION, kernel_initializer=config.KERNEL_INITIALIZER, kernel_regularizer=config.KERNEL_REGULARIZER)(vgg16_output)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = config.POOLING(x)      
    x = Dense(128,activation=config.HIDDEN_ACTIVATION, kernel_initializer=config.KERNEL_INITIALIZER, kernel_regularizer=config.KERNEL_REGULARIZER)(x)
    x = Dropout(0.3)(x)
    x = Dense(64,activation=config.HIDDEN_ACTIVATION, kernel_initializer=config.KERNEL_INITIALIZER, kernel_regularizer=config.KERNEL_REGULARIZER)(x)
    x = Dropout(0.3)(x)
    
    vgg16_output = Dense(38,activation=config.OUTPUT_LYR_ACTIVATION, kernel_initializer=config.KERNEL_INITIALIZER, kernel_regularizer=config.KERNEL_REGULARIZER)(x)
    
    model = Model(inputs=[input_to_vgg16], outputs=[vgg16_output])

    optimizer = config.OPTIMIZER(config.EPSILON)
    model.compile(optimizer=optimizer, loss=config.LOSS_FN, metrics=['accuracy'])
    return model

def training():
    model = plant_disease_cnn()
    print(model.summary())
    batch_size = config.BATCH_SIZE
    training_data = data_augmentation.data_augmentation()
    history = model.fit(
    training_data,
    validation_data=data_loading.validation_set,
    epochs=config.EPOCHS,  # Adjust the number of epochs based on your dataset
    steps_per_epoch=len(data_loading.TRAIN_DF) // batch_size,
    validation_steps=len(data_loading.VALID_DF) // batch_size)
    
    return history

# if __name__ == "__main__":
#     print(training())