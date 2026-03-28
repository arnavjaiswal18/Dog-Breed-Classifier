import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# app/main.py

from fastapi import FastAPI, UploadFile, File
import shutil
from .predict import predict_image

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(
    title="Dog Breed Classifier API",
    description="Upload a dog image and get predicted breed",
    version="1.0"
)

# -----------------------------
# Create upload folder
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "🚀 Dog Breed Classifier API is running"
    }

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run prediction
        breed, confidence = predict_image(file_path)

        # Remove file after prediction
        os.remove(file_path)

        return {
            "success": True,
            "breed": breed,
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }