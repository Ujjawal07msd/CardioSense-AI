import unittest
import json
import io
import pandas as pd
from app import app, MODEL_FILE
import os

class HeartDiseaseAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_model_exists(self):
        self.assertTrue(os.path.exists(MODEL_FILE), "heart_model.pkl should exist")

    def test_index_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'CardioSense AI', response.data)
        self.assertIn(b'Patient Clinical Parameters', response.data)

    def test_predict_normal_patient(self):
        normal_patient = {
            'age': 67, 'sex': 1, 'cp': 0, 'trestbps': 120, 'chol': 229,
            'fbs': 0, 'restecg': 0, 'thalach': 129, 'exang': 1,
            'oldpeak': 2.6, 'slope': 1, 'ca': 2, 'thal': 3
        }
        response = self.app.post('/predict', data=json.dumps(normal_patient), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['prediction'], 0)
        self.assertEqual(data['label'], 'Low Risk / Normal')
        print("Normal Patient Test Result:", data['label'], f"Risk: {data['risk_score']}%")

    def test_predict_high_risk_patient(self):
        risk_patient = {
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }
        response = self.app.post('/predict', data=json.dumps(risk_patient), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['prediction'], 1)
        self.assertEqual(data['label'], 'Heart Disease Detected')
        print("High Risk Patient Test Result:", data['label'], f"Risk: {data['risk_score']}%", f"Flags: {data['risk_factors']}")

    def test_predict_batch_csv(self):
        df_sample = pd.read_csv('heart.csv').head(5).drop(columns=['target'])
        csv_bytes = io.BytesIO()
        df_sample.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)
        
        response = self.app.post('/predict_batch', 
                                data={'file': (csv_bytes, 'test_patients.csv')},
                                content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get('Content-Type'), 'text/csv; charset=utf-8')
        result_df = pd.read_csv(io.BytesIO(response.data))
        self.assertIn('Prediction', result_df.columns)
        self.assertIn('Heart_Disease_Risk_%', result_df.columns)
        self.assertEqual(len(result_df), 5)
        print("Batch CSV Prediction Test Result: Successfully generated predictions for 5 rows")

if __name__ == '__main__':
    unittest.main()
