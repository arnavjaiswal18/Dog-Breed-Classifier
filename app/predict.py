# app/predict.py

import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ------------------------------------
# Setup Paths (Absolute - no working directory dependency)
# ------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "weights" / "model.h5"
CLASS_NAMES_PATH = Path(__file__).resolve().parent / "class_names.json"

# Lazy-load model (only when predict_image is called)
model = None

import json
# Load class names from JSON
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)


def get_model():
    """Lazy-load model to avoid failures at import time"""
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Train the model first."
            )
        model = load_model(str(MODEL_PATH))
    return model


# ------------------------------------
# Prediction Function
# ------------------------------------
def predict_image(img_path):
    """
    Predict dog breed from image path
    Returns: (breed, confidence)
    """
    loaded_model = get_model()

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = loaded_model.predict(img_array)

    # Get result
    index = np.argmax(predictions)
    breed = CLASS_NAMES[index]
    confidence = float(np.max(predictions))

    return breed, confidence


# ------------------------------------
# Test Block (run directly)
# ------------------------------------
if __name__ == "__main__":
    import glob
    test_images = glob.glob(str(PROJECT_ROOT / "data" / "test" / "*.jpg"))
    
    if test_images:
        test_image = test_images[0]
        breed, confidence = predict_image(test_image)
        print(f"🐶 Prediction: {breed}")
        print(f"📊 Confidence: {confidence:.2f}")
    else:
        print("❌ No test images found!")