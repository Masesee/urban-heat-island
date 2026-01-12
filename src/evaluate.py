import pandas as pd
import numpy as np
import argparse
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
import os

def evaluate(model_path, input_path, output_dir='reports', cv_folds=5):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Preprocess
    subset = ['median_NDVI', 'median_NDBI', 'median_NDWI', 'building_density_100m']
    df = df.drop_duplicates(subset=subset)
    
    features = ['median_NDVI', 'median_NDBI', 'median_NDWI', 'building_density_100m']
    target = 'UHI_Class'
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in input data.")
        
    X = df[features].values
    y = df[target].values
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Load scaler
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')
    if not os.path.exists(scaler_path):
         scaler_path = model_path + '_scaler.pkl'
         
    if os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        print("Warning: Scaler not found. Using raw features.")
        X_scaled = X
        
    # Load Encoder
    encoder_path = model_path.replace('.pkl', '_encoder.pkl')
    if not os.path.exists(encoder_path):
         encoder_path = model_path + '_encoder.pkl'
    
    if os.path.exists(encoder_path):
        print(f"Loading encoder from {encoder_path}...")
        le = joblib.load(encoder_path)
        # Encode y to match model output for scoring if needed, 
        # OR decode model output to match y.
        # Since y is string, and model predicts int, let's decode model output for confusion matrix.
        # But for cross_val_score, we need X and y to match. 
        # Model expects X_scaled and returns int. 
        # So we should encode y to int for cross_val_score.
        y_encoded = le.transform(y)
    else:
        print("Warning: Encoder not found. Assuming y is already numeric or model outputs strings.")
        le = None
        y_encoded = y # Fallback
        
    # Create report directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(output_dir, f'evaluation_{timestamp}')
    os.makedirs(report_dir, exist_ok=True)
    print(f"Saving reports to {report_dir}...")

    # Cross Validation
    print(f"Performing {cv_folds}-fold Cross Validation...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    # Use y_encoded for CV
    scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    
    print(f"CV Accuracy Scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Save CV results
    with open(os.path.join(report_dir, 'cv_results.txt'), 'w') as f:
        f.write(f"CV Accuracy Scores: {scores}\n")
        f.write(f"Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})\n")

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        print("Calculating Feature Importance...")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print and Save Feature Importance
        print("Feature Ranking:")
        feature_importance_df = pd.DataFrame({
            'Feature': [features[i] for i in indices],
            'Importance': importances[indices]
        })
        print(feature_importance_df)
        feature_importance_df.to_csv(os.path.join(report_dir, 'feature_importance.csv'), index=False)
        
        # Plot Feature Importance
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'feature_importance.png'))
        plt.close()
    
    # Confusion Matrix
    print("Generating Confusion Matrix (on full dataset)...")
    y_pred = model.predict(X_scaled)
    
    if le:
        # Decode predictions to strings for report
        y_pred_labels = le.inverse_transform(y_pred)
        # y is already strings
        report_y_true = y
        report_y_pred = y_pred_labels
        labels = le.classes_
    else:
        report_y_true = y
        report_y_pred = y_pred
        labels = model.classes_
    
    report = classification_report(report_y_true, report_y_pred)
    print("Classification Report:")
    print(report)
    
    with open(os.path.join(report_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    cm = confusion_matrix(report_y_true, report_y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    
    # Save plot
    plot_path = os.path.join(report_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UHI Model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--input", required=True, help="Path to input CSV (with target)")
    parser.add_argument("--output", default="reports", help="Path to output directory for reports")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    
    args = parser.parse_args()
    
    evaluate(args.model, args.input, args.output, args.cv)
