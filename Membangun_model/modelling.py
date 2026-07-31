"""
modelling.py - Training Model ML dengan MLflow Autolog
Proyek Akhir MSML - Alvin Rama Saputra

Kriteria 2 (Basic): Melatih model menggunakan MLflow autolog
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

def load_preprocessed_data(data_dir="diabetes_preprocessing"):
    """Load dataset yang sudah dipreproses"""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_model():
    """Training model dengan MLflow autolog"""
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Aktifkan MLflow autolog
    mlflow.sklearn.autolog()
    
    # Enable file store backend for newer MLflow version
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    
    # Set experiment dan tracking URI lokal
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Diabetes_Prediction")
    
    with mlflow.start_run(run_name="GradientBoosting_autolog"):
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n=== Hasil Evaluasi Model ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        print(f"\nModel berhasil dilatih dan tersimpan di MLflow!")
        print(f"Jalankan 'mlflow ui' untuk melihat dashboard.")

if __name__ == "__main__":
    train_model()
