"""
automate_Alvin-Rama-Saputra.py
Script otomasi preprocessing dataset Diabetes
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    """Load dataset dari file CSV"""
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(filepath, names=columns)
    # Pima indians dataset doesn't have headers in the raw csv, but if it does we can skip setting names or use header=0
    # The dataset I downloaded might not have headers, but if it does, the above might make it a row. 
    # Let's handle it safely by checking the first row.
    if df.iloc[0]['Pregnancies'] == '6' or df.iloc[0]['Pregnancies'] == 6:
        pass # No header in data
    elif type(df.iloc[0]['Pregnancies']) == str and not df.iloc[0]['Pregnancies'].isdigit():
        df = df.iloc[1:].reset_index(drop=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
    return df

def handle_missing_values(df):
    """Handle missing values (0 → median) pada kolom tertentu"""
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df

def split_data(df, target_col='Outcome', test_size=0.2):
    """Split data menjadi train/test"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scaling fitur menggunakan StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_test_scaled

def preprocess_pipeline(input_path, output_dir):
    """Pipeline preprocessing lengkap"""
    print(f"Loading data from {input_path}...")
    df = load_data(input_path)
    
    print("Handling missing values...")
    df = handle_missing_values(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    print("Scaling features...")
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Simpan hasil
    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"Preprocessing selesai. Data disimpan di '{output_dir}'.")

if __name__ == "__main__":
    # Path dataset dan output disesuaikan dengan struktur folder
    preprocess_pipeline(
        input_path="../diabetes.csv",
        output_dir="diabetes_preprocessing"
    )
