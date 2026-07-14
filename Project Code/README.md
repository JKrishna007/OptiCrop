# 🌾 OptiCrop — Smart Agricultural Production Optimization Engine

OptiCrop is a machine-learning-powered crop recommendation system that helps farmers, researchers, and policymakers make data-driven agricultural decisions. It analyzes soil nutrients and environmental conditions to recommend the optimal crop for maximum yield and sustainability.

---

## ✨ Features

| Feature | Description |
|---|---|
| **ML-Powered Predictions** | Compares 5 algorithms and uses the best-performing model (Random Forest) |
| **7 Input Parameters** | Nitrogen, Phosphorous, Potassium, Temperature, Humidity, pH, Rainfall |
| **22 Crop Recommendations** | Rice, Maize, Cotton, Coffee, Banana, Mango, and 16 more |
| **REST API** | Programmatic access via `/api/predict` endpoint |
| **Premium Web UI** | Dark-themed glassmorphism design with micro-animations |
| **Quick-Fill Samples** | One-click sample data for fast testing |
| **Growing Tips** | Crop-specific advice on seasons and cultivation practices |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask
- **ML**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib (.pkl)

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <repository-url>
cd OptiCrop

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Train Model

```bash
# Generate synthetic dataset (2,200 samples)
python generate_dataset.py

# Train ML models and save the best one
python -m ml.train
```

### 4. Run the Application

```bash
python app.py
```

Visit **http://127.0.0.1:5000** in your browser.

---

## 📡 API Usage

```bash
GET /api/predict?N=80&P=45&K=45&temperature=24&humidity=85&ph=6.5&rainfall=220
```

**Response:**
```json
{
  "crop": "rice",
  "confidence": 99.2,
  "emoji": "🌾",
  "season": "Kharif",
  "tip": "Requires standing water; ideal in clay-loam soils with good water retention.",
  "input_params": { "N": 80, "P": 45, "K": 45, "temperature": 24, "humidity": 85, "ph": 6.5, "rainfall": 220 }
}
```

---

## 📂 Project Structure

```
OptiCrop/
├── app.py                      # Flask application
├── config.py                   # Configuration
├── generate_dataset.py         # Synthetic data generator
├── requirements.txt            # Dependencies
├── data/
│   └── crop_recommendation.csv # Dataset
├── models/
│   ├── model.pkl               # Trained ML model
│   ├── scaler.pkl              # Feature scaler
│   └── label_encoder.pkl       # Label encoder
├── ml/
│   ├── __init__.py
│   ├── preprocess.py           # Data preprocessing
│   ├── train.py                # Model training & evaluation
│   └── predict.py              # Prediction pipeline
├── static/
│   ├── css/style.css           # Premium dark theme
│   └── js/main.js              # Client-side interactivity
└── templates/
    ├── base.html               # Base layout
    ├── index.html              # Home / input form
    ├── result.html             # Prediction result
    └── about.html              # Methodology & docs
```

---

## 🤖 Machine Learning Models

| Algorithm | Type | Key Strength |
|---|---|---|
| K-Nearest Neighbors | Instance-based | Simple, effective for small datasets |
| Logistic Regression | Linear classifier | Interpretable probability estimates |
| Decision Tree | Tree-based | Captures non-linear patterns |
| **Random Forest** ⭐ | Ensemble | Best accuracy (~99%), robust to overfitting |
| Naive Bayes | Probabilistic | Very fast baseline classifier |

---

## 🌱 Supported Crops

Rice • Maize • Chickpea • Kidney Beans • Pigeon Peas • Moth Beans • Mung Bean • Black Gram • Lentil • Pomegranate • Banana • Mango • Grapes • Watermelon • Muskmelon • Apple • Orange • Papaya • Coconut • Cotton • Jute • Coffee

---

## 📄 License

This project is for educational and research purposes.

---

Built with ❤️ for sustainable agriculture.
