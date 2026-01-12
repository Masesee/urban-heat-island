# EY Urban Heat Island Challenge

## 🌍 Project Overview
This project aims to build a classification model to predict **Urban Heat Island (UHI) intensity** (Low, Medium, High) for a new city (Sierra Leone). The model is trained using ground-truth temperature points, Sentinel-2 satellite imagery, and building-footprint data from Brazil and Chile.

For a detailed, step-by-step guide on the project lifecycle, please refer to the [Walkthrough](walkthrough.md).

## 📂 Repository Structure

```text
EY_UHI_Challenge/
├── data/                 # Raw and processed datasets (Brazil, Chile, Sierra Leone)
├── notebooks/            # Jupyter notebooks for analysis and modeling
├── src/                  # Modularized source code
│   ├── model_arch/       # Model architecture definitions
│   ├── evaluate.py       # Script for model evaluation
│   ├── predict.py        # Script for generating predictions
│   ├── train.py          # Script for training models
│   └── utils.py          # Shared utility functions (feature extraction)
├── models/               # Directory for saved models (timestamped subfolders)
├── submissions/          # Generated submission files (timestamped subfolders)
├── reports/              # Generated evaluation reports (timestamped subfolders)
├── walkthrough.md        # Detailed project walkthrough and guide
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Lab or Notebook

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "EY Urban Heat Island Challenge"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Usage

The project has been modularized into standalone scripts in the `src/` directory.

### 1. Training
Train a model using `src/train.py`. You can choose from multiple architectures (Random Forest, XGBoost, etc.).
```bash
python src/train.py --input data/uhi_data.csv --model-out models/rf_model.pkl --model-arch random_forest
```

### 2. Prediction
Generate predictions for new data using `src/predict.py`. Supports raw data (extracts features automatically) or pre-processed data.
```bash
# Using pre-processed features and original ID file
python src/predict.py --input data/test_features.csv --model models/train_.../rf_model.pkl --output submissions/predictions.csv --id-file data/Test.csv
```

### 3. Evaluation
Evaluate model performance using `src/evaluate.py`. Generates CV results, confusion matrices, and feature importance plots.
```bash
python src/evaluate.py --model models/train_.../rf_model.pkl --input data/uhi_data.csv --output reports
```

## 📊 Data Overview
You can download all the data from the competition page: [Zindi Urban Heat Island Challenge Data](https://zindi.africa/competitions/urban-heat-island-challenge/data)

The `data/` directory contains the following key files:
- **Training Data:** `sample_brazil_uhi_data.csv`, `sample_chile_uhi_data.csv` (Lat/Lon with UHI Class)
- **Test Data:** `Validation_Dataset.csv` (Lat/Lon to predict), `test_data_sierra.csv`
- **Satellite Imagery:** `sample_Brazil.tiff`, `sample_chile.tiff`, `sample_Sierra.tiff` (Sentinel-2 GeoTIFFs)
- **Building Footprints:** Shapefiles for Brazil, Chile, and Sierra Leone.

## 📓 Notebooks
- `Sample_GeoTiff_Creation.ipynb`: Guide on processing GeoTIFF data.
- `Sample_Median_Mosaic.ipynb`: Creating median mosaics from satellite imagery.
- `Sample_Model_Notebook.ipynb`: Baseline model implementation.

## 🤝 Contributing
Please follow the guidelines in the [Walkthrough](walkthrough.md) for coding style and version control practices.
