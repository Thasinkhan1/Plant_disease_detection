from keras.regularizers import l2 # type: ignore
from keras.layers import  GlobalAveragePooling2D # type: ignore
from keras.optimizers import SGD,Adam # type: ignore
from keras.applications import VGG16 # type: ignore
import os


POOLING = GlobalAveragePooling2D()



BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_PATH, "Dataset/train")
VALID_PATH = os.path.join(BASE_PATH, "Dataset/valid")
TEST_PATH = os.path.join(BASE_PATH, "Dataset/test")
DATA_DIR = "Dataset"

SAVED_MODEL_FILE = "model.h5"
SAVED_MODEL_PATH = "models/model.h5"
IMAGE_PATH_FOR_TESTING = "Dataset/test/AppleCedarRust1.JPG"

EPSILON = 0.001

COLOR_RGB = "rgb"

INTERPOLATION = "bilinear"

BATCH_SIZE = 32

EPOCHS = 10
IMAGE_SIZE = (128, 128)

KERNEL_SIZE = (3, 3)

OPTIMIZER = Adam

LOSS_FN = "categorical_crossentropy"

INPUT_SHAPE = (128, 128, 3)

INCLUDE_TOP = False

MODEL = VGG16  

KERNEL_INITIALIZER = "he_normal"

KERNEL_REGULARIZER = l2(0.01) 

HIDDEN_ACTIVATION = "relu"

OUTPUT_LYR_ACTIVATION = "softmax"

APP_BG_IMG = "models/img1.jpeg"


CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", 
    "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", 
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", 
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", 
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", 
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", 
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", 
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

ABOUT_APP = """
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """

APP_MARKDOWN = """
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """