from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import tempfile
from app.predict import predict_image

# Create FastAPI app
app = FastAPI(title="Dog Breed Classifier")

# Serve static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Serve the main React app
@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)


# API endpoint for predictions
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and predict dog breed"""
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Predict breed and confidence
        breed, confidence = predict_image(tmp_path)

        # Delete temp file
        import os
        os.remove(tmp_path)

        return {
            "breed": breed,
            "confidence": f"{confidence:.2%}"
        }

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}