# 🐶 Dog Breed Classifier

A FastAPI web application that uses deep learning to classify dog breeds from images. Powered by transfer learning with MobileNetV2, this classifier can identify 120 different dog breeds with high accuracy.

## 🌐 Live Demo

**Try it now!** → [dog-breed-classifier-yup9.onrender.com](https://dog-breed-classifier-yup9.onrender.com)

Simply upload a dog image and get instant breed predictions!

## 📋 Features

- **120 Dog Breed Classification** - Recognizes a wide variety of dog breeds
- **FastAPI REST API** - Efficient, modern Python API framework
- **Transfer Learning** - Uses pretrained MobileNetV2 model for optimal performance
- **React Web Interface** - Interactive frontend for image upload and prediction
- **Image Upload** - Supports single image uploads for real-time predictions
- **Docker Support** - Containerized for easy deployment
- **High Performance** - Lightweight model suitable for cloud deployment
- **Data Augmentation** - Training includes rotation, zoom, and horizontal flip

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, TensorFlow/Keras
- **Frontend**: React 18, HTML5
- **Model**: MobileNetV2 (Transfer Learning)
- **Data Processing**: Pillow, NumPy
- **Deployment**: Docker, Gunicorn
- **Python Version**: 3.11.9

## 📦 Requirements

```
fastapi>=0.104.0
uvicorn>=0.23.2
python-multipart>=0.0.6
pillow>=10.1.0
tensorflow-cpu>=2.16.1
numpy<2.0.0
gunicorn
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11.9
- pip or conda
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Dog-Breed-Classifier
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and prepare data**
   - Place training images in `data/train/`
   - Ensure `data/labels.csv` contains image IDs and breed labels

5. **Train the model (optional)**
   ```bash
   python model/train.py
   ```

6. **Run the application**
   ```bash
   python -m uvicorn app.ui_main:app --host 0.0.0.0 --port 8000
   ```

   Or with reload for development:
   ```bash
   python -m uvicorn app.ui_main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Access the web interface**
   - Open browser and navigate to `http://localhost:8000`
   - Or use the live demo: [https://dog-breed-classifier-yup9.onrender.com](https://dog-breed-classifier-yup9.onrender.com)

## 🐳 Docker Deployment

### Build the Docker Image
```bash
docker build -t dog-breed-classifier .
```

### Run the Container
```bash
docker run -p 8000:10000 dog-breed-classifier
```

### Deploy to Cloud (Render, Heroku, etc.)
The application is configured to use the PORT environment variable:
```bash
docker run -p 8000:10000 -e PORT=10000 dog-breed-classifier
```

## 📚 API Documentation

### Interactive API Docs
Once running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints

#### GET `/`
Returns the home page with the React interface.

#### GET `/health`
Health check endpoint.
```json
{
  "status": "ok"
}
```

#### POST `/predict`
Upload an image to get breed prediction.

**Request:**
- `Content-Type`: `multipart/form-data`
- `file`: Image file (.jpg, .png, etc.)

**Response:**
```json
{
  "breed": "golden_retriever",
  "confidence": "98.45%"
}
```

## 🧠 Model Architecture

The classifier uses **MobileNetV2** with transfer learning:

1. **Base Model**: Pretrained MobileNetV2 (ImageNet weights)
2. **Custom Layers**:
   - Global Average Pooling 2D
   - Dense layer with 256 units + ReLU activation
   - Output layer with 120 units + Softmax activation

**Key Benefits**:
- Lightweight (~45MB) - suitable for mobile and cloud deployment
- Fast inference - real-time predictions
- High accuracy - pretrained on ImageNet with 1.4M images
- Transfer learning - reduces training time and data requirements

## 📊 Training

### Training Configuration
- **Input Size**: 224×224 RGB images
- **Batch Size**: 32
- **Epochs**: 10 (with early stopping)
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy

### Data Augmentation
- Rotation range: 20°
- Zoom range: 20%
- Horizontal flip: Enabled

### Train/Validation Split
- Training: 80%
- Validation: 20%
- Stratified by breed to maintain class balance

**To retrain the model:**
```bash
python model/train.py
```

The trained model saves as `model/weights/model.h5`

## 📁 Project Structure

```
Dog-Breed-Classifier/
├── app/
│   ├── __init__.py
│   ├── main.py              # REST API endpoints
│   ├── ui_main.py           # FastAPI app with frontend
│   ├── predict.py           # Prediction function
│   ├── class_names.json     # 120 breed class names
│   ├── static/
│   │   └── index.html       # React frontend
│   └── templates/
│       └── index.html       # Alternative HTML template
├── model/
│   ├── __init__.py
│   ├── model.py             # Model architecture
│   ├── train.py             # Training script
│   └── weights/
│       └── model.h5         # Trained model weights
├── data/
│   ├── labels.csv           # Image ID to breed mapping
│   ├── train/               # Original training images
│   ├── train_split/         # Organized training data (120 breeds)
│   ├── valid/               # Validation data
│   ├── test/                # Test images
│   └── sample_submission.csv
├── utils/
│   └── preprocessing.py     # Data preprocessing & splitting
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python version
└── README.md               # This file
```

## 🐕 Supported Dog Breeds (120 Total)

The classifier supports 120 dog breeds including:
- Affenpinscher
- Afghan Hound
- Beagle
- Border Collie
- Boxer
- Chihuahua
- Dalmatian
- French Bulldog
- German Shepherd
- Golden Retriever
- Labrador Retriever
- Poodle (all sizes)
- Rottweiler
- Siberian Husky
- Yorkshire Terrier
- And 105 more!

See [app/class_names.json](app/class_names.json) for the complete list.

## 🎯 Performance

- **Model Size**: ~45MB (weights)
- **Inference Time**: ~500ms per image (CPU)
- **Accuracy**: Designed for high accuracy on diverse dog images
- **Supported Image Formats**: JPG, PNG, and other common formats

## 📝 Example Usage

### Python Script
```python
from app.predict import predict_image

breed, confidence = predict_image("path/to/dog_image.jpg")
print(f"Breed: {breed}")
print(f"Confidence: {confidence:.2%}")
```

### Via cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/dog_image.jpg"
```

### Via Python Requests
```python
import requests

with open("dog_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

## 🐛 Troubleshooting

### Model file not found
- Ensure `model/weights/model.h5` exists
- Run `python model/train.py` to train the model first

### Import errors with TensorFlow
- Reinstall TensorFlow: `pip install --upgrade tensorflow-cpu`
- Use Python 3.11.x as specified in `runtime.txt`

### Image upload issues
- Verify `uploads/` directory exists
- Check file permissions
- Ensure image format is supported (JPG, PNG)

### Port already in use
- Change the port: `python -m uvicorn app.ui_main:app --port 8001`

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 👨‍💼 Contact

For questions or feedback, please open an issue on the repository.

---

**Happy classifying! 🐶**