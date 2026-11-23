# CardioSense AI — Heart Disease Prediction System

[![Live Demo](https://img.shields.io/badge/Live_Demo-CardioSense_AI-brightgreen?style=for-the-badge&logo=render)](https://cardiosense-ai-gpw4.onrender.com/)
[![Interactive Docs](https://img.shields.io/badge/Swagger_UI-/docs-teal?style=for-the-badge&logo=swagger)](https://cardiosense-ai-gpw4.onrender.com/docs)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Validation-Pydantic_v2-e92063?style=for-the-badge&logo=pydantic)](https://docs.pydantic.dev/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

> 🌐 **Live Web Application:** [https://cardiosense-ai-gpw4.onrender.com/](https://cardiosense-ai-gpw4.onrender.com/)  
> 📖 **Interactive Swagger API Documentation:** [https://cardiosense-ai-gpw4.onrender.com/docs](https://cardiosense-ai-gpw4.onrender.com/docs)

A complete, end-to-end Machine Learning web application and diagnostic support tool to predict the presence or risk of heart disease in patients using clinical biomarkers.

Based on the UCI Heart Disease Dataset, evaluating multiple classification models (Random Forest, Gradient Boosting, Logistic Regression, etc.).

---

## 🌟 Key Features

1. **Machine Learning Model Engine (`train_model.py`)**:
   - Automated model evaluation across algorithms.
   - Evaluates Random Forest, Gradient Boosting, and Logistic Regression with Stratified Cross-Validation.
   - Saves model and metadata (`heart_model.pkl`) with high accuracy and probability confidence.

2. **Modern Interactive Web Application (`app.py` & `templates/index.html`)**:
   - Clean, responsive clinical dashboard with real-time risk assessment.
   - Dynamic Risk Score progress bar and risk severity flags (Hypertension, High Cholesterol, ST depression, Angina).
   - Instant 1-click patient preset autofill buttons (**Sample: Normal** and **Sample: High Risk**) for instant testing.
   - Clinical lifestyle and medical recommendations tailored to the prediction outcome.

3. **Batch Prediction via CSV (`/predict_batch`)**:
   - Upload any CSV with patient records matching the 13 feature columns.
   - Automatically computes risk predictions and exports a downloadable CSV with predicted classifications and probabilities.

4. **REST API Endpoint (`/predict`)**:
   - Standard JSON API endpoint for integration into external apps, hospital systems, or mobile clients.

---

## 📋 Clinical Features Used

| Feature | Name | Description | Values / Range |
|---|---|---|---|
| `age` | Age | Patient age in years | 29 – 77 |
| `sex` | Sex | Biological sex | `1` = Male, `0` = Female |
| `cp` | Chest Pain Type | Constriction/pain experience | `0`: Typical Angina, `1`: Atypical Angina, `2`: Non-anginal, `3`: Asymptomatic |
| `trestbps` | Resting BP | Resting blood pressure | mm Hg (e.g., 94 – 200) |
| `chol` | Cholesterol | Serum cholesterol | mg/dl (e.g., 126 – 564) |
| `fbs` | Fasting Blood Sugar | Sugar level > 120 mg/dl | `1` = True, `0` = False |
| `restecg` | Resting ECG | Resting electrocardiogram | `0`: Normal, `1`: ST-T wave abnormality, `2`: Left ventricular hypertrophy |
| `thalach` | Max Heart Rate | Maximum heart rate achieved | bpm (e.g., 71 – 202) |
| `exang` | Exercise Angina | Exercise-induced angina | `1` = Yes, `0` = No |
| `oldpeak` | ST Depression | ST depression induced by exercise | Numeric (e.g., 0.0 – 6.2) |
| `slope` | ST Slope | Slope of peak exercise ST segment | `0`: Upsloping, `1`: Flat, `2`: Downsloping |
| `ca` | Major Vessels | Number of major vessels (0–3) | 0, 1, 2, 3 |
| `thal` | Thalassemia | Blood disorder test | `1` = Normal, `2` = Fixed defect, `3` = Reversible defect |

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model
```bash
python train_model.py
```
This evaluates the models and outputs `heart_model.pkl`.

### 3. Run the FastAPI Application
```bash
python app.py
```
*(Or `uvicorn app:app --reload --port 5000`)*

Open your browser and navigate to:
- **Web Dashboard:** `http://127.0.0.1:5000`
- **Interactive Swagger Docs:** `http://127.0.0.1:5000/docs`

Or simply double-click **`run.bat`** on Windows!

---

## 📡 REST API Usage

### Single Patient Prediction:
**`POST /predict`**
```json
{
  "age": 60,
  "sex": 1,
  "cp": 0,
  "trestbps": 145,
  "chol": 282,
  "fbs": 0,
  "restecg": 0,
  "thalach": 142,
  "exang": 1,
  "oldpeak": 2.8,
  "slope": 1,
  "ca": 2,
  "thal": 3
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": 1,
  "label": "Heart Disease Detected",
  "confidence": 92.0,
  "risk_score": 92.0,
  "risk_factors": [
    "High resting blood pressure (Hypertension Stage 2)",
    "High serum cholesterol (Hypercholesterolemia)",
    "Significant ST segment depression on exercise",
    "Presence of exercise-induced angina",
    "Fluoroscopy detects vessel blockage"
  ],
  "recommendations": [...]
}
```
