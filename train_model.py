import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def train():
    data_path = os.path.join(os.path.dirname(__file__), 'heart.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # We evaluate several algorithms
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    best_score = 0
    best_model_name = None
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Model: {name} -> Accuracy: {acc * 100:.2f}%")
        if acc > best_score:
            best_score = acc
            best_model_name = name
            best_model = model

    print(f"\nSelected Best Model: {best_model_name} with {best_score * 100:.2f}% accuracy")
    
    # Save model and feature names
    save_data = {
        'model': best_model,
        'model_name': best_model_name,
        'feature_names': list(X.columns),
        'accuracy': round(best_score * 100, 2)
    }
    
    model_path = os.path.join(os.path.dirname(__file__), 'heart_model.pkl')
    joblib.dump(save_data, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train()

