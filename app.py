import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
import io

app = Flask(__name__)

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

@app.route('/')
def index():
    global model, feature_names, model_accuracy, model_name
    if model is None:
        try:
            model, feature_names, model_accuracy, model_name = load_or_train_model()
        except Exception as e:
            print(f"Error loading model: {e}")
    return render_template('index.html', accuracy=model_accuracy, model_name=model_name)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        model, _, _, _ = load_or_train_model()
        
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
        
        # Expected features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        input_data = [
            float(data.get('age', 50)),
            float(data.get('sex', 1)),
            float(data.get('cp', 0)),
            float(data.get('trestbps', 120)),
            float(data.get('chol', 200)),
            float(data.get('fbs', 0)),
            float(data.get('restecg', 0)),
            float(data.get('thalach', 150)),
            float(data.get('exang', 0)),
            float(data.get('oldpeak', 0.0)),
            float(data.get('slope', 1)),
            float(data.get('ca', 0)),
            float(data.get('thal', 2))
        ]
        
        features_df = pd.DataFrame([input_data], columns=feature_names if feature_names else [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])
        prediction = int(model.predict(features_df)[0])
        
        # Calculate risk probability if model supports predict_proba
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_df)[0]
            confidence = round(float(probs[prediction]) * 100, 2)
            risk_score = round(float(probs[1]) * 100, 2)
        else:
            confidence = 100.0
            risk_score = 100.0 if prediction == 1 else 0.0

        # High risk factors assessment
        risk_factors = []
        if float(data.get('trestbps', 120)) >= 140:
            risk_factors.append("High resting blood pressure (Hypertension Stage 2)")
        if float(data.get('chol', 200)) >= 240:
            risk_factors.append("High serum cholesterol (Hypercholesterolemia)")
        if float(data.get('thalach', 150)) < 110:
            risk_factors.append("Low maximum exercise heart rate")
        if float(data.get('oldpeak', 0)) >= 2.0:
            risk_factors.append("Significant ST segment depression on exercise")
        if float(data.get('exang', 0)) == 1:
            risk_factors.append("Presence of exercise-induced angina")
        if float(data.get('ca', 0)) >= 1:
            risk_factors.append("Fluoroscopy detects vessel blockage")

        result = {
            'status': 'success',
            'prediction': prediction,
            'label': 'Heart Disease Detected' if prediction == 1 else 'Low Risk / Normal',
            'confidence': confidence,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': [
                "Schedule a comprehensive cardiovascular checkup with an authorized physician.",
                "Maintain a low-sodium, heart-healthy Mediterranean or DASH diet.",
                "Engage in 150 minutes of moderate aerobic activity weekly as recommended by AHA.",
                "Regularly monitor your resting blood pressure and lipid profile."
            ] if prediction == 1 else [
                "Continue maintaining a healthy lifestyle, balanced diet, and regular exercise routine.",
                "Perform annual health checkups and blood pressure screenings.",
                "Keep cholesterol and fasting glucose levels within standard healthy ranges."
            ]
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    global model
    if model is None:
        model, _, _, _ = load_or_train_model()
        
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'status': 'error', 'message': 'Please upload a .csv file'}), 400
        
    try:
        df = pd.read_csv(file)
        required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return jsonify({'status': 'error', 'message': f'Missing columns in CSV: {", ".join(missing)}'}), 400
            
        X_input = df[required_cols]
        preds = model.predict(X_input)
        df['Prediction'] = ['Heart Disease Detected' if p == 1 else 'Normal' for p in preds]
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_input)
            df['Heart_Disease_Risk_%'] = [round(p[1] * 100, 1) for p in probs]
            
        out_csv = io.BytesIO()
        df.to_csv(out_csv, index=False)
        out_csv.seek(0)
        
        return send_file(
            out_csv,
            mimetype='text/csv',
            as_attachment=True,
            download_name='heart_disease_predictions.csv'
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Heart Disease Prediction Server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

