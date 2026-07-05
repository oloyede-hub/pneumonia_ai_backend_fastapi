# 🫁 Pneumonia Detection Backend API

A FastAPI backend that powers an AI-driven chest X-ray pneumonia detection application. The API performs image preprocessing, deep learning inference using a TensorFlow/Keras model, generates Grad-CAM visualizations for explainability, and stores uploaded images securely using Supabase Storage.

The backend is designed to work with a Next.js frontend, providing prediction results, confidence scores, explainable AI outputs, and image assets through a clean REST API.

---

## ✨ Features

* FastAPI REST API
* TensorFlow/Keras model inference
* Chest X-ray preprocessing with OpenCV
* Binary classification (Normal vs Pneumonia)
* Grad-CAM heatmap generation
* Explainable AI (XAI) metadata
* Heatmap statistics and activation analysis
* Supabase Storage integration
* CORS support for frontend applications
* JSON API responses

---

## 🧠 Explainable AI

Beyond predicting whether pneumonia is present, the API explains the model's decision using Grad-CAM.

For every prediction, the backend generates:

* Prediction label
* Confidence score
* Class probabilities
* Grad-CAM overlay image
* Raw Grad-CAM heatmap
* Interpolated heatmap surface
* Top activation regions
* Heatmap statistics

This enables downstream applications to visualize and interpret the model's reasoning.

---

## 🏗️ Tech Stack

### Framework

* FastAPI
* Uvicorn

### Machine Learning

* TensorFlow / Keras
* VGG16 Transfer Learning
* Grad-CAM

### Image Processing

* OpenCV
* NumPy

### Cloud Storage

* Supabase Storage

### Other

* Python
* python-dotenv

---

## 📁 Project Structure

```text
backend/
│
├── model/
│   └── pneumonia.keras
│
├── routes/
│
├── services/
│   ├── predictor.py
│   ├── gradcam.py
│   └── supabase_storage.py
│
├── utils/
│   └── storage_utils.py
│
├── uploads/
│
├── static/
│
├── main.py
├── requirements.txt
└── .env
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/oloyede-hub/pneumonia_backend.git
```

Navigate into the project:

```bash
cd pneumonia-backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## ⚙️ Environment Variables

Create a `.env` file with the following values:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_SECRET_KEY=your_supabase_service_role_key
```

---

## 📡 API Response

Example prediction response:

```json
{
  "prediction": {
    "label": "Pneumonia",
    "confidence": 99.98,
    "probability": {
      "Normal": 0.0002,
      "Pneumonia": 0.9998
    }
  },
  "images": {
    "original_url": "...",
    "gradcam_url": "..."
  },
  "gradcam": {
    "surface": {
      "width": 224,
      "height": 224,
      "values": []
    },
    "raw": {
      "width": 14,
      "height": 14,
      "values": []
    },
    "analysis": {
      "threshold": 0.6,
      "activatedPercentage": 3.57,
      "topActivations": []
    },
    "statistics": {
      "maxActivation": 0.99,
      "minActivation": 0.0,
      "meanActivation": 0.09,
      "stdActivation": 0.17
    }
  }
}
```

---

## 🔄 Backend Workflow

```text
Client Upload
      │
      ▼
FastAPI Endpoint
      │
      ▼
Image Preprocessing
(OpenCV)
      │
      ▼
TensorFlow Model
      │
      ▼
Prediction
      │
      ├────────► Probability Scores
      │
      ├────────► Grad-CAM Generation
      │
      ├────────► Heatmap Analysis
      │
      └────────► Upload Images to Supabase Storage
                     │
                     ▼
              JSON Response
```

---

## 📊 Model Performance

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 93.1% |
| Precision |   93% |
| Recall    |   96% |
| F1 Score  |   95% |
| ROC AUC   | 0.978 |

---

## 🔮 Future Improvements

* Multi-class thoracic disease detection
* DICOM image support
* ONNX/TensorRT model optimization
* Batch image inference
* Authentication and authorization
* Prediction history
* PDF diagnostic reports
* Docker deployment
* Automated testing
* CI/CD pipeline

---

## 🤝 Frontend

This backend is designed to work with a Next.js frontend that provides:

* Chest X-ray upload
* Prediction dashboard
* Grad-CAM overlay visualization
* Interactive 3D activation surface
* Activation statistics
* Explainable AI interface

---

## 👨‍💻 Author

**Sodiq Oloyede**

* GitHub: https://github.com/oloyede-hub

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
