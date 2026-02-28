import os 
import uvicorn
import numpy as np # type: ignore
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from PIL import Image # type: ignore
from io import BytesIO
from api.predictors import predict
from config import config
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from data_transformation.data_augmentation import clean_class_name

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://127.0.0.1:8000"] to be more strict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict")
async def predict_disease(request: Request, file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    preds = predict(image)
    class_idx = int(np.argmax(preds))
    class_name = config.CLASS_NAMES[class_idx]
    confidence = float(np.max(preds)) * 100
    #advice = config.ADVICE[class_name]
    readable_name = clean_class_name(class_name)
    
    if "healthy" in readable_name.lower():
        return JSONResponse({
            "status": "healthy",
            "disease_name": "Healthy",
            "confidence": round(confidence, 2),
            "message": "The plant appears healthy."
        })

    else:
        advice = config.ADVICE.get(class_name, {
            "care": "No care info available.",
            "prevention": "No prevention info available.",
            "organic_medicine": "Consult expert.",
            "chemical_medicine": "Consult expert.",
            "precautions": "Follow safety guidelines."
        })

        return JSONResponse({
            "status": "diseased",
            "disease_name": readable_name,
            "confidence": round(confidence, 2),
            "advice": advice
        })
    
@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)