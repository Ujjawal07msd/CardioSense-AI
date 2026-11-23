import unittest
import io
import pandas as pd
from fastapi.testclient import TestClient
from app import app, MODEL_FILE
import os

class HeartDiseaseFastAPITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_model_exists(self):
        self.assertTrue(os.path.exists(MODEL_FILE), "heart_model.pkl should exist")

    def test_index_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('CardioSense AI', response.text)
        self.assertIn('Patient Clinical Parameters', response.text)
        print("PASS: Web Dashboard (GET /) loaded successfully")

    def test_docs_pages(self):
        docs_res = self.client.get('/docs')
        self.assertEqual(docs_res.status_code, 200)
        redoc_res = self.client.get('/redoc')
        self.assertEqual(redoc_res.status_code, 200)
        print("PASS: OpenAPI Swagger (/docs) and ReDoc (/redoc) accessible")

    def test_predict_normal_patient(self):
        normal_patient = {
            'age': 67, 'sex': 1, 'cp': 0, 'trestbps': 120.0, 'chol': 229.0,
            'fbs': 0, 'restecg': 0, 'thalach': 129.0, 'exang': 1,
            'oldpeak': 2.6, 'slope': 1, 'ca': 2, 'thal': 3
        }
        response = self.client.post('/predict', json=normal_patient)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['prediction'], 0)
        self.assertEqual(data['label'], 'Low Risk / Normal')
        print(f"PASS: Normal Patient prediction -> {data['label']} (Risk: {data['risk_score']}%)")

    def test_predict_high_risk_patient(self):
        risk_patient = {
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145.0, 'chol': 233.0,
            'fbs': 1, 'restecg': 0, 'thalach': 150.0, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }
        response = self.client.post('/predict', json=risk_patient)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['prediction'], 1)
        self.assertEqual(data['label'], 'Heart Disease Detected')
        self.assertTrue(len(data['risk_factors']) > 0)
        print(f"PASS: High Risk Patient prediction -> {data['label']} (Risk: {data['risk_score']}%)")

    def test_validation_error(self):
        # Age negative or beyond 120 should be caught by Pydantic automatically
        invalid_patient = {
            'age': -10, 'sex': 1, 'cp': 0, 'trestbps': 120.0, 'chol': 229.0,
            'fbs': 0, 'restecg': 0, 'thalach': 129.0, 'exang': 1,
            'oldpeak': 2.6, 'slope': 1, 'ca': 2, 'thal': 3
        }
        response = self.client.post('/predict', json=invalid_patient)
        self.assertEqual(response.status_code, 422)  # Pydantic validation error code
        print("PASS: Pydantic input validation correctly rejected invalid age (-10)")

    def test_predict_batch_csv(self):
        df_sample = pd.read_csv('heart.csv').head(5).drop(columns=['target'])
        csv_bytes = io.BytesIO()
        df_sample.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)

        response = self.client.post(
            '/predict_batch',
            files={'file': ('test_patients.csv', csv_bytes, 'text/csv')}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/csv', response.headers.get('content-type', ''))
        result_df = pd.read_csv(io.BytesIO(response.content))
        self.assertIn('Prediction', result_df.columns)
        self.assertIn('Heart_Disease_Risk_%', result_df.columns)
        self.assertEqual(len(result_df), 5)
        print("PASS: Batch CSV predictions successfully generated for 5 patients")

if __name__ == '__main__':
    unittest.main()
