"""
modelling_tuning.py - Training Model ML dengan Hyperparameter Tuning
Proyek Akhir MSML - Alvin Rama Saputra

Kriteria 2 (Skilled): Manual logging + hyperparameter tuning
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os

def load_preprocessed_data(data_dir="diabetes_preprocessing"):
    """Load dataset yang sudah dipreproses"""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_with_tuning():
    # Enable file store backend for newer MLflow version
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Diabetes_Prediction")
    
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Hyperparameter combinations
    param_grid = [
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 7, "learning_rate": 0.01},
    ]
    
    for i, params in enumerate(param_grid):
        with mlflow.start_run(run_name=f"GradientBoosting_tuning_{i+1}"):
            # Manual log params
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("max_depth", params["max_depth"])
            mlflow.log_param("learning_rate", params["learning_rate"])
            mlflow.log_param("random_state", 42)
            
            model = GradientBoostingClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            # Manual log metrics (sama dengan yang dicatat autolog)
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted', zero_division=0))
            mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted', zero_division=0))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted', zero_division=0))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log confusion matrix sebagai artefak
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ax.matshow(cm, cmap='Blues')
            for (j, k), val in np.ndenumerate(cm):
                ax.text(k, j, str(val), ha='center', va='center')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - Run {i+1}')
            cm_path = f"confusion_matrix_run_{i+1}.png"
            fig.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)
            
            print(f"Run {i+1} dengan parameter {params} selesai!")

if __name__ == "__main__":
    train_with_tuning()
