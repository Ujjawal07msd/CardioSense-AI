@echo off
echo ========================================================
echo CardioSense AI - Heart Disease Prediction System (FastAPI)
echo ========================================================
echo.

echo [1/3] Checking dependencies...
pip install -r requirements.txt

echo.
echo [2/3] Checking Machine Learning model...
if not exist heart_model.pkl (
    echo Model not found. Training model now...
    python train_model.py
)

echo.
echo [3/3] Starting FastAPI Web Server...
echo Web Dashboard: http://127.0.0.1:5000
echo Interactive API Docs: http://127.0.0.1:5000/docs
echo.
python -m uvicorn app:app --host 127.0.0.1 --port 5000 --reload

pause
