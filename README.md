# EY Urban Heat Island Challenge

## 🌍 Project Overview
This project aims to build a classification model to predict **Urban Heat Island (UHI) intensity** (Low, Medium, High) for a new city (Sierra Leone). The model is trained using ground-truth temperature points, Sentinel-2 satellite imagery, and building-footprint data from Brazil and Chile.

For a detailed, step-by-step guide on the project lifecycle, please refer to the [Walkthrough](walkthough.md).

## 📂 Repository Structure

```text
EY_UHI_Challenge/
├── data/                 # Raw and processed datasets (Brazil, Chile, Sierra Leone)
├── notebooks/            # Jupyter notebooks for analysis and modeling
├── src/                  # Source code (currently empty, intended for scripts)
├── models/               # Directory for saved models
├── submissions/          # Generated submission files
├── walkthough.md         # Detailed project walkthrough and guide
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

## 📊 Data Overview
The `data/` directory contains the following key files:
- **Training Data:** `sample_brazil_uhi_data.csv`, `sample_chile_uhi_data.csv` (Lat/Lon with UHI Class)
- **Test Data:** `Validation_Dataset.csv` (Lat/Lon to predict), `test_data_sierra.csv`
- **Satellite Imagery:** `sample_Brazil.tiff`, `sample_chile.tiff`, `sample_Sierra.tiff` (Sentinel-2 GeoTIFFs)
- **Building Footprints:** Shapefiles for Brazil, Chile, and Sierra Leone.

## 📓 Notebooks
- `Sample_GeoTiff_Creation.ipynb`: Guide on processing GeoTIFF data.
- `Sample_Median_Mosaic.ipynb`: Creating median mosaics from satellite imagery.
- `Sample_Model_Notebook.ipynb`: Baseline model implementation.
- `notebook_v01.ipynb`: Experimental notebook.

## 🤝 Contributing
Please follow the guidelines in the [Walkthrough](walkthough.md) for coding style and version control practices.
