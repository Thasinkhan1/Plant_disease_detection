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
                
ADVICE = {
    "Apple___Apple_scab": {
        "check": "Olive-green or brown velvety spots on leaves or fruit.",
        "advice": "Use scab-resistant varieties, remove fallen leaves, and apply fungicide early in spring."
    },
    "Apple___Black_rot": {
        "check": "Circular black spots on fruit with red halo.",
        "advice": "Prune dead branches, avoid fruit injury, and burn infected debris."
    },
    "Apple___Cedar_apple_rust": {
        "check": "Yellow-orange spots on upper leaf surfaces.",
        "advice": "Remove nearby cedar trees and apply fungicides early."
    },
    "Apple___healthy": {
        "check": "No visible spots or lesions on fruit or leaves.",
        "advice": "Maintain hygiene, proper pruning, and routine monitoring."
    },
    "Blueberry___healthy": {
        "check": "Glossy green leaves, no discoloration.",
        "advice": "Use well-drained acidic soil and water properly."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "check": "White powdery coating on leaves.",
        "advice": "Avoid overhead watering, prune infected areas, and use sulfur-based sprays."
    },
    "Cherry_(including_sour)___healthy": {
        "check": "Leaves are shiny and uniformly green.",
        "advice": "Mulch and monitor especially during bloom season."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "check": "Narrow brown lesions on leaves.",
        "advice": "Use resistant hybrids and apply foliar fungicides."
    },
    "Corn_(maize)___Common_rust_": {
        "check": "Reddish-brown pustules on leaves.",
        "advice": "Avoid overhead irrigation and use rust-resistant varieties."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "check": "Elliptical gray-green lesions on leaves.",
        "advice": "Practice crop rotation and use hybrid seeds."
    },
    "Corn_(maize)___healthy": {
        "check": "Bright green upright leaves.",
        "advice": "Use nitrogen-rich fertilizer and inspect weekly."
    },
    "Grape___Black_rot": {
        "check": "Brown circular leaf spots and black shriveled grapes.",
        "advice": "Prune vines and apply early fungicides."
    },
    "Grape___Esca_(Black_Measles)": {
        "check": "Tiger-striped leaves, shriveled berries.",
        "advice": "Remove infected vines and avoid over-fertilizing."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "check": "Angular brown spots on leaves.",
        "advice": "Improve air circulation and spray fungicides post-bloom."
    },
    "Grape___healthy": {
        "check": "Broad green leaves and healthy grapes.",
        "advice": "Keep vines pruned and sunlight adequate."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "check": "Yellow shoots and asymmetrical mottled leaves.",
        "advice": "Remove infected trees and control psyllids with insecticides."
    },
    "Peach___Bacterial_spot": {
        "check": "Water-soaked lesions on leaves/fruit.",
        "advice": "Use copper-based sprays and avoid wet foliage."
    },
    "Peach___healthy": {
        "check": "No spots on fruits or leaves.",
        "advice": "Thin fruits and apply dormant sprays."
    },
    "Pepper,_bell___Bacterial_spot": {
        "check": "Greasy-looking spots on leaves/fruit.",
        "advice": "Use certified seeds and apply copper fungicides."
    },
    "Pepper,_bell___healthy": {
        "check": "Firm leaves and uniform growth.",
        "advice": "Maintain moisture and proper spacing."
    },
    "Potato___Early_blight": {
        "check": "Bullseye-shaped spots on older leaves.",
        "advice": "Remove infected debris and rotate crops annually."
    },
    "Potato___Late_blight": {
        "check": "Dark lesions and fuzzy mold under leaves.",
        "advice": "Use certified seeds and apply fungicides before rains."
    },
    "Potato___healthy": {
        "check": "Strong, upright stems and healthy leaves.",
        "advice": "Use mulch and avoid overwatering."
    },
    "Raspberry___healthy": {
        "check": "Vibrant green leaves and no cane damage.",
        "advice": "Ensure well-drained soil and prune dead canes."
    },
    "Soybean___healthy": {
        "check": "Uniform green foliage.",
        "advice": "Rotate crops and inspect for pests weekly."
    },
    "Squash___Powdery_mildew": {
        "check": "White powder on leaf surfaces.",
        "advice": "Use neem oil or sulfur sprays and water at base."
    },
    "Strawberry___Leaf_scorch": {
        "check": "Purple spots with dry centers.",
        "advice": "Avoid wet leaves and crowding; rotate plants."
    },
    "Strawberry___healthy": {
        "check": "Green glossy leaves and healthy flowers.",
        "advice": "Keep sunlight ample and mulch around plants."
    },
    "Tomato___Bacterial_spot": {
        "check": "Dark greasy spots on leaves and fruit.",
        "advice": "Use drip irrigation and copper sprays."
    },
    "Tomato___Early_blight": {
        "check": "Target-like spots on lower leaves.",
        "advice": "Remove infected leaves and rotate crops."
    },
    "Tomato___Late_blight": {
        "check": "Black lesions and fuzzy mold.",
        "advice": "Spray fungicides before rain and prune affected parts."
    },
    "Tomato___Leaf_Mold": {
        "check": "Yellow spots above, mold below leaves.",
        "advice": "Use sulfur sprays and prune lower leaves."
    },
    "Tomato___Septoria_leaf_spot": {
        "check": "Small round gray spots with dark borders.",
        "advice": "Use mulch and avoid water splash on leaves."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "check": "Yellow speckled leaves and fine webbing.",
        "advice": "Use insecticidal soap or neem oil; mist leaves occasionally."
    },
    "Tomato___Target_Spot": {
        "check": "Large concentric lesions with yellow halo.",
        "advice": "Avoid overhead watering and prune lower leaves."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "check": "Curled yellow leaves and stunted growth.",
        "advice": "Control whiteflies and remove infected plants."
    },
    "Tomato___Tomato_mosaic_virus": {
        "check": "Mosaic mottling on leaves.",
        "advice": "Sanitize tools and avoid infected seedlings."
    },
    "Tomato___healthy": {
        "check": "Deep green leaves and healthy fruit.",
        "advice": "Fertilize regularly and maintain good airflow."
    }
}
