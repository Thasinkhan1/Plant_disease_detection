import numpy as np # type: ignore
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from PIL import Image # type: ignore
from io import BytesIO
from project_root.fastapi.predictors import predict
from project_root.config import config
from keras.models import load_model # type: ignore
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://127.0.0.1:8000"] to be more strict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="project_root/static"), name="static")
templates = Jinja2Templates(directory="project_root/templates")

# @app.on_event("startup")
# def startup_event():
#load_model = load_model(config.SAVED_MODEL_PATH)
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
    #advice = config.ADVICE[class_name]
    advice = config.ADVICE.get(class_name, {
        "check": "No advice available.",
        "advice": "Please consult an expert."
    })
    return JSONResponse({
        "prediction": class_name,
        "confidence": f"{confidence:.2f}",
        "advice": advice
    })