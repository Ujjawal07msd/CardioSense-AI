@echo off
echo ===================================================
echo CardioSense AI - Heart Disease Prediction System
echo ===================================================
echo.

echo [1/3] Checking dependencies...
pip install -r requirements.txt

echo.
echo [2/3] Training and checking Machine Learning model...
python train_model.py

echo.
echo [3/3] Starting Web Application...
echo Open your browser at: http://127.0.0.1:5000
python app.py

pause

