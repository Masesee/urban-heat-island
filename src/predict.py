import pandas as pd
import argparse
import joblib
import os
import sys

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

def predict(input_path, model_path, output_path, tiff_path=None, shp_path=None, id_file=None):
    print(f"Loading input data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Check for features
    features = ['median_NDVI', 'median_NDBI', 'median_NDWI', 'building_density_100m']
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"Missing features: {missing_features}. Attempting extraction...")
        if not tiff_path or not shp_path:
            # Check if we are missing specific ones
            missing_spectral = any(f in missing_features for f in ['median_NDVI', 'median_NDBI', 'median_NDWI'])
            missing_density = 'building_density_100m' in missing_features
            
            if missing_spectral and not tiff_path:
                 raise ValueError("Spectral features are missing. Please provide --tiff argument.")
            if missing_density and not shp_path:
                 raise ValueError("Building density feature is missing. Please provide --shp argument.")
            
        # Extract spectral indices
        if any(f in missing_features for f in ['median_NDVI', 'median_NDBI', 'median_NDWI']):
            print("Extracting spectral indices...")
            df = utils.extract_band_values(df, tiff_path)
            
        # Compute building density
        if 'building_density_100m' in missing_features:
            print("Computing building density...")
            df = utils.compute_building_density(df, shp_path)
    
    # Load model and scaler
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')
    if not os.path.exists(scaler_path):
         # Try appending
         scaler_path = model_path + '_scaler.pkl'
         
    if os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
        X = scaler.transform(df[features].values)
    else:
        print("Warning: Scaler not found. Using raw features (this may lead to poor performance if the model expects scaled data).")
        X = df[features].values
        
    # Load Encoder
    encoder_path = model_path.replace('.pkl', '_encoder.pkl')
    if not os.path.exists(encoder_path):
         encoder_path = model_path + '_encoder.pkl'
    
    if os.path.exists(encoder_path):
        print(f"Loading encoder from {encoder_path}...")
        le = joblib.load(encoder_path)
    else:
        print("Warning: Encoder not found. Predictions will be numeric.")
        le = None
        
    # Predict
    print("Generating predictions...")
    predictions = model.predict(X)
    
    if le:
        predictions = le.inverse_transform(predictions)
    
    # Prepare submission dataframe
    ids = None
    if id_file:
        print(f"Loading IDs from {id_file}...")
        id_df = pd.read_csv(id_file)
        if 'ID' in id_df.columns:
            ids = id_df['ID']
        else:
            print(f"Warning: 'ID' column not found in {id_file}.")
            
    if ids is None and 'ID' in df.columns:
        ids = df['ID']
        
    if ids is not None:
        if len(ids) != len(predictions):
             print(f"Warning: Length of IDs ({len(ids)}) does not match length of predictions ({len(predictions)}).")
             # Fallback to just saving predictions if lengths don't match, or truncate/pad?
             # For safety, let's just warn and try to proceed if possible, or fail.
             # Assuming strict matching for submission.
        
        submission_df = pd.DataFrame({
            'ID': ids,
            'Target': predictions
        })
    else:
        print("Warning: 'ID' column not found in input or id-file. Saving all columns with 'Target'.")
        df['Target'] = predictions
        submission_df = df
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create timestamped directory in submissions
    # Assuming output_path is something like "submissions/predictions.csv"
    # We want "submissions/predict_2023.../predictions.csv"
    
    base_dir = os.path.dirname(output_path)
    if not base_dir:
        base_dir = 'submissions'
        
    # Extract model name from model_path for folder naming
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    run_dir = os.path.join(base_dir, f'{model_name}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    filename = os.path.basename(output_path)
    if not filename:
        filename = 'predictions.csv'
        
    final_output_path = os.path.join(run_dir, filename)
    
    submission_df.to_csv(final_output_path, index=False)
    print(f"Predictions saved to {final_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict UHI Class")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--output", required=True, help="Path to save predictions CSV")
    parser.add_argument("--tiff", help="Path to Sentinel-2 GeoTIFF (required if spectral features missing)")
    parser.add_argument("--shp", help="Path to Building Footprints Shapefile (required if building density missing)")
    parser.add_argument("--id-file", help="Path to original CSV containing ID column (optional)")
    
    args = parser.parse_args()
    
    predict(args.input, args.model, args.output, args.tiff, args.shp, args.id_file)
