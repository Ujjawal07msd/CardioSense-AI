import os
import io
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Initialize FastAPI with OpenAPI documentation
app = FastAPI(
    title="CardioSense AI — Heart Disease Prediction API",
    description="Production-grade Machine Learning API and Clinical Decision Support System for cardiovascular risk evaluation.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup Jinja2 Templates
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
templates = Jinja2Templates(directory=templates_dir)

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'heart_model.pkl')

def load_or_train_model():
    if not os.path.exists(MODEL_FILE):
        print("Model file not found. Training model now...")
        from train_model import train
        train()
    data = joblib.load(MODEL_FILE)
    return data['model'], data.get('feature_names', []), data.get('accuracy', 0), data.get('model_name', 'Model')

model, feature_names, model_accuracy, model_name = None, [], 0, ''
try:
    model, feature_names, model_accuracy, model_name = load_or_train_model()
except Exception as e:
    print(f"Warning: Model could not be loaded immediately: {e}")


# Pydantic Schema for Patient Biomarkers
class PatientInput(BaseModel):
    age: int = Field(52, ge=1, le=120, description="Age in years (1 - 120)")
    sex: int = Field(1, ge=0, le=1, description="Biological sex (1 = Male, 0 = Female)")
    cp: int = Field(0, ge=0, le=3, description="Chest pain type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)")
    trestbps: float = Field(125.0, ge=50, le=250, description="Resting blood pressure in mm Hg")
    chol: float = Field(212.0, ge=100, le=600, description="Serum cholesterol in mg/dl")
    fbs: int = Field(0, ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)")
    restecg: int = Field(0, ge=0, le=2, description="Resting ECG results (0: Normal, 1: ST-T wave abnormality, 2: LV hypertrophy)")
    thalach: float = Field(168.0, ge=50, le=250, description="Maximum heart rate achieved in bpm")
    exang: int = Field(0, ge=0, le=1, description="Exercise induced angina (1 = Yes, 0 = No)")
    oldpeak: float = Field(1.0, ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest")
    slope: int = Field(1, ge=0, le=2, description="Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)")
    ca: int = Field(0, ge=0, le=4, description="Number of major vessels (0 - 3, occasionally 4) colored by fluoroscopy")
    thal: int = Field(2, ge=0, le=3, description="Thalassemia (1 = Normal, 2 = Fixed defect, 3 = Reversible defect)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145.0,
                "chol": 233.0,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150.0,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }


class PredictionResponse(BaseModel):
    status: str
    prediction: int
    label: str
    confidence: float
    risk_score: float
    risk_factors: List[str]
    recommendations: List[str]


@app.get("/", response_class=HTMLResponse, summary="Web Dashboard")
async def home(request: Request):
    """Render the CardioSense AI interactive web dashboard."""
    global model, feature_names, model_accuracy, model_name
    if model is None:
        model, feature_names, model_accuracy, model_name = load_or_train_model()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "accuracy": model_accuracy,
            "model_name": model_name
        }
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict Heart Disease Risk")
async def predict(patient: PatientInput):
    """
    Evaluate 13 clinical biomarkers using Machine Learning to predict cardiovascular disease risk.
    """
    global model, feature_names
    if model is None:
        model, feature_names, _, _ = load_or_train_model()

    try:
        # Convert Pydantic object to DataFrame with matching column names
        input_dict = patient.model_dump()
        features_df = pd.DataFrame([input_dict])
        
        # Ensure column ordering matches training data
        if feature_names:
            features_df = features_df[feature_names]

        prediction = int(model.predict(features_df)[0])

        confidence = 0.0
        risk_score = 0.0
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_df)[0]
            confidence = round(float(probs[prediction]) * 100, 2)
            risk_score = round(float(probs[1]) * 100, 2)
        else:
            confidence = 100.0
            risk_score = 100.0 if prediction == 1 else 0.0

        # High risk factors assessment
        risk_factors = []
        if patient.trestbps >= 140:
            risk_factors.append("High resting blood pressure (Hypertension Stage 2)")
        if patient.chol >= 240:
            risk_factors.append("High serum cholesterol (Hypercholesterolemia)")
        if patient.thalach < 110:
            risk_factors.append("Low maximum exercise heart rate")
        if patient.oldpeak >= 2.0:
            risk_factors.append("Significant ST segment depression on exercise")
        if patient.exang == 1:
            risk_factors.append("Presence of exercise-induced angina")
        if patient.ca >= 1:
            risk_factors.append("Fluoroscopy detects vessel blockage")

        recommendations = [
            "Schedule a comprehensive cardiovascular checkup with an authorized physician.",
            "Maintain a low-sodium, heart-healthy Mediterranean or DASH diet.",
            "Engage in 150 minutes of moderate aerobic activity weekly as recommended by AHA.",
            "Regularly monitor your resting blood pressure and lipid profile."
        ] if prediction == 1 else [
            "Continue maintaining a healthy lifestyle, balanced diet, and regular exercise routine.",
            "Perform annual health checkups and blood pressure screenings.",
            "Keep cholesterol and fasting glucose levels within standard healthy ranges."
        ]

        return PredictionResponse(
            status="success",
            prediction=prediction,
            label="Heart Disease Detected" if prediction == 1 else "Low Risk / Normal",
            confidence=confidence,
            risk_score=risk_score,
            risk_factors=risk_factors,
            recommendations=recommendations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", summary="Batch Predict from CSV File")
async def predict_batch(file: UploadFile = File(...)):
    """
    Upload a CSV file containing multiple patient biomarker records and receive predictions.
    """
    global model, feature_names
    if model is None:
        model, feature_names, _, _ = load_or_train_model()

    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a .csv file."
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns in CSV: {', '.join(missing)}"
            )

        X_input = df[required_cols]
        preds = model.predict(X_input)
        df['Prediction'] = ['Heart Disease Detected' if p == 1 else 'Normal' for p in preds]

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_input)
            df['Heart_Disease_Risk_%'] = [round(p[1] * 100, 1) for p in probs]

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=heart_disease_predictions.csv"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting CardioSense AI FastAPI Server on http://0.0.0.0:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
