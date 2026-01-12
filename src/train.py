import pandas as pd
import numpy as np
import argparse
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import importlib

def load_model_arch(model_arch_name):
    """
    Dynamically imports the model architecture module.
    """
    try:
        module = importlib.import_module(f"model_arch.{model_arch_name}")
        return module.get_model
    except ImportError:
        raise ValueError(f"Model architecture '{model_arch_name}' not found in src/model_arch/")
    except AttributeError:
        raise ValueError(f"Module 'src/model_arch/{model_arch_name}.py' must have a 'get_model' function.")

def train(input_path, model_out_path, model_arch_name, test_size=0.3, random_state=42):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Preprocess: Drop duplicates based on features
    print("Preprocessing data...")
    subset = ['median_NDVI', 'median_NDBI', 'median_NDWI', 'building_density_100m']
    initial_len = len(df)
    df = df.drop_duplicates(subset=subset)
    print(f"Dropped {initial_len - len(df)} duplicates.")
    
    # Select features and target
    features = ['median_NDVI', 'median_NDBI', 'median_NDWI', 'building_density_100m']
    target = 'UHI_Class'
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in input data.")
        
    X = df[features].values
    y = df[target].values
    
    # Encode Target
    print("Encoding target labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {le.classes_}")
    
    # Split data
    print(f"Splitting data (test_size={test_size})...")
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Load Model Architecture
    print(f"Loading model architecture: {model_arch_name}...")
    get_model = load_model_arch(model_arch_name)
    model = get_model(random_state=random_state)
    
    # Train model
    print(f"Training {model.__class__.__name__}...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("Evaluating on validation set...")
    val_preds = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, val_preds, target_names=le.classes_))
    
    # Save artifacts
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = os.path.dirname(model_out_path)
    if not base_dir:
        base_dir = 'models'
        
    run_dir = os.path.join(base_dir, f'train_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    filename = os.path.basename(model_out_path)
    if not filename:
        filename = 'model.pkl'
        
    final_model_path = os.path.join(run_dir, filename)
    joblib.dump(model, final_model_path)
    
    scaler_path = os.path.join(run_dir, filename.replace('.pkl', '_scaler.pkl'))
    if scaler_path == final_model_path:
         scaler_path = final_model_path + '_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    
    encoder_path = os.path.join(run_dir, filename.replace('.pkl', '_encoder.pkl'))
    if encoder_path == final_model_path:
        encoder_path = final_model_path + '_encoder.pkl'
    joblib.dump(le, encoder_path)
    
    print(f"Model saved to {final_model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Encoder saved to {encoder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UHI Model")
    parser.add_argument("--input", required=True, help="Path to input CSV (must contain features and target)")
    parser.add_argument("--model-out", required=True, help="Path to save the trained model (e.g., models/rf_model.pkl)")
    parser.add_argument("--model-arch", required=True, help="Name of the model architecture module in src/model_arch (e.g., random_forest)")
    parser.add_argument("--test-size", type=float, default=0.3, help="Fraction of data to use for validation")
    
    args = parser.parse_args()
    
    train(args.input, args.model_out, args.model_arch, args.test_size)
