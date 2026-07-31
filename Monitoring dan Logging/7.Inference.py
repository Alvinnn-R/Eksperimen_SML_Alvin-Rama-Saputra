"""
Inference.py - Script Inference/Serving Model ML
Proyek Akhir MSML - Alvin Rama Saputra
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import json

# Pastikan MLflow membaca folder mlruns yang ada di Membangun_model
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
mlflow.set_tracking_uri("file:../Membangun_model/mlruns")

def find_latest_run(experiment_name="Diabetes_Prediction"):
    """Cari run terakhir dari experiment MLflow"""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' tidak ditemukan.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise ValueError("Tidak ada run yang ditemukan.")
    return runs[0]

import pickle

def load_model(run_id=None):
    """Load model langsung via pickle untuk menghindari bug metadata MLflow"""
    model_path = f"../Membangun_model/mlruns/605703438274022565/{run_id}/artifacts/model/model.pkl"
    print(f"Loading model dari: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    return model

def predict(model, input_data):
    """Lakukan prediksi"""
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    results = []
    for i in range(len(predictions)):
        result = {
            "prediction": int(predictions[i]),
            "label": "Diabetes" if predictions[i] == 1 else "Normal",
            "confidence": float(max(probabilities[i])),
            "probabilities": {
                "Normal": float(probabilities[i][0]),
                "Diabetes": float(probabilities[i][1])
            }
        }
        results.append(result)
    return results

def main():
    print("=" * 60)
    print("Diabetes Prediction - Inference")
    print("=" * 60)
    
    # Menggunakan Run ID secara langsung untuk menghindari bug MLflow di Windows
    model = load_model("84f5469954754c3ea1594aef17381cf8")
    
    data_dir = "../Membangun_model/diabetes_preprocessing"
    if not os.path.exists(data_dir):
        data_dir = "diabetes_preprocessing"
    
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    sample_data = X_test.head(5)
    
    print(f"\nInput data ({len(sample_data)} sampel):")
    print(sample_data.to_string())
    
    results = predict(model, sample_data)
    
    print(f"\n{'='*60}")
    print("Hasil Prediksi:")
    print(f"{'='*60}")
    for i, result in enumerate(results):
        print(f"\nSampel {i+1}:")
        print(f"  Prediksi   : {result['label']} ({result['prediction']})")
        print(f"  Confidence : {result['confidence']:.4f}")
        print(f"  Prob Normal   : {result['probabilities']['Normal']:.4f}")
        print(f"  Prob Diabetes : {result['probabilities']['Diabetes']:.4f}")
    
    output_file = "inference_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nHasil disimpan ke: {output_file}")

if __name__ == "__main__":
    main()
