import numpy as np # type: ignore
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from PIL import Image # type: ignore
from io import BytesIO
from project_root.fastapi.predictors import predict
from project_root.config import config
from keras.models import load_model # type: ignore

app = FastAPI()

app.mount("/static", StaticFiles(directory="project_root/static"), name="static")
templates = Jinja2Templates(directory="project_root/templates")

# @app.on_event("startup")
# def startup_event():
load_model = load_model(config.SAVED_MODEL_PATH)
#     load_model()
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_disease(request: Request, file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    preds = predict(image)
    class_idx = int(np.argmax(preds))
    class_name = config.CLASS_NAMES[class_idx]
    confidence = float(np.max(preds)) * 100
    advice = config.ADVICE[class_name]
    
    return {
        "prediction is: ": class_name,
        "with confidence: ": f"{confidence:.2f}",
        "advice: ": advice
    }
